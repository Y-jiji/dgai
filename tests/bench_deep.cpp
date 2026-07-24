// DEEP-200M DGAI bench: two-phase windowed insert+search protocol, matching
// fnct-db's hermes_sift.rs and PipeANN's tests/bench_loop.cpp so all three
// systems produce comparable logs. DGAI's DynamicSSDIndex (v2/dynamic_index.h)
// refuses to attach unless "<prefix>_disk.index" already exists, so phase 0
// bootstraps a small on-disk graph offline via build_disk_index; those
// bootstrap points never go through insert_in_place and get no per-point row.
//
// Phase 1 (BOOTSTRAP->BASE): plain sequential insert, no search.
// Phase 2 (BASE->N): each BATCH-point step runs one insert thread
// concurrently with a fixed-query search stream (undersampled logging --
// the only undersampled step), then a fully-logged (k,L,beam) search sweep
// at the same just-reached corpus size. Recall is computed post-hoc against
// a ground-truth log, so rows carry the returned neighbor ids/dists.

#include "v2/dynamic_index.h"
#include "aux_utils.h"
#include "distance.h"
#include "global_stats.h"
#include "utils.h"

#include <omp.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using T = float;
using TagT = uint32_t;
using clk = std::chrono::steady_clock;

constexpr size_t DIM = 96;
constexpr size_t N = 200'000'000;
constexpr size_t BASE = 100'000'000;
constexpr size_t BATCH = 1'000'000;
constexpr size_t QLEN = 10'000;
// Built offline via build_disk_index; never goes through insert_in_place.
constexpr size_t BOOTSTRAP = 100'000;

constexpr unsigned BUILD_R = 32;
constexpr unsigned BUILD_L = 64;
constexpr double BUILD_B_GB = 1.0;
constexpr double BUILD_M_GB = 8.0;
constexpr unsigned BUILD_THREADS = 32;

constexpr unsigned R_DISK = 32;
constexpr unsigned L_DISK = 100;
constexpr float ALPHA_DISK = 1.2f;
constexpr unsigned C_DISK = 160;
constexpr unsigned BEAMWIDTH = 4;
// DGAI's DRAM node-cache knob (SSDIndex::cache_bfs_levels/load_cache_list).
// Kept as a documented constant for whoever tunes RSS on nezha: this
// codebase's DynamicSSDIndex ctor never calls cache_bfs_levels, so the
// Parameters entry below is otherwise a dead read; wiring the cache up
// ourselves was skipped since a BFS-levels snapshot cache would go stale
// under this workload's concurrent inserts without extra invalidation work.
constexpr unsigned NODES_TO_CACHE = 0;

constexpr unsigned SEARCH_THREADS = 8;
constexpr unsigned NUM_THREADS = 32;

constexpr unsigned K = 10;
// Undersample rate for the phase-2 concurrent insert+search step only.
constexpr unsigned MIXRATE = 50;

struct SearchCfg {
  unsigned k, l, beam;
};
constexpr SearchCfg PRIMARY_CFG{10, 100, BEAMWIDTH};
const std::vector<SearchCfg> EXTRA_CFGS = {
    {10, 50, BEAMWIDTH},
    {10, 150, BEAMWIDTH},
    {10, 200, BEAMWIDTH},
    {10, 100, 8},
};

void read_raw(const std::string &path, size_t off, size_t count, size_t dim, std::vector<T> &out) {
  out.assign(count * dim, T{});
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "open " << path << "\n";
    std::exit(1);
  }
  f.seekg(static_cast<std::streamoff>(off * dim * sizeof(T)), std::ios::beg);
  f.read(reinterpret_cast<char *>(out.data()), count * dim * sizeof(T));
  if (!f) {
    std::cerr << "short read " << path << " off=" << off << " count=" << count << "\n";
    std::exit(1);
  }
}

void write_bootstrap_bin(const std::string &data_path, const std::string &out_path, size_t n, size_t dim) {
  std::vector<T> buf;
  read_raw(data_path, 0, n, dim, buf);
  std::ofstream w(out_path, std::ios::binary);
  int npts_i32 = (int) n, dim_i32 = (int) dim;
  w.write((char *) &npts_i32, sizeof(int));
  w.write((char *) &dim_i32, sizeof(int));
  w.write((char *) buf.data(), buf.size() * sizeof(T));
}

void write_insert_header(std::ofstream &of) {
  of << "phase,batch,tag,start_ns,lat_ns,n_ios,io_us\n";
}

void write_insert_row(std::ofstream &of, std::mutex &mu, const char *phase, uint64_t batch, TagT tag,
                      uint64_t start_ns, uint64_t lat_ns, uint64_t n_ios, double io_us) {
  std::ostringstream row;
  row << phase << ',' << batch << ',' << tag << ',' << start_ns << ',' << lat_ns << ',' << n_ios << ',' << io_us
      << '\n';
  std::lock_guard<std::mutex> lock(mu);
  of << row.str();
}

void write_query_header(std::ofstream &of) {
  of << "batch,visitor,qi,start_ns,lat_ns";
  for (unsigned i = 1; i <= K; i++) of << ",id@" << i << ",dist@" << i;
  of << ",n_ios,n_hops,n_cmps,total_us\n";
}

void write_query_row(std::ofstream &of, std::mutex &mu, uint64_t batch, const std::string &visitor, size_t qi,
                     uint64_t start_ns, uint64_t lat_ns, const TagT *ids, const float *dists,
                     const pipeann::QueryStats &stats) {
  std::ostringstream row;
  row << batch << ',' << visitor << ',' << qi << ',' << start_ns << ',' << lat_ns;
  for (unsigned i = 0; i < K; i++) row << ',' << ids[i] << ',' << dists[i];
  row << ',' << (uint64_t) stats.n_ios << ',' << (uint64_t) stats.n_hops << ',' << (uint64_t) stats.n_cmps << ','
      << stats.total_us << '\n';
  std::lock_guard<std::mutex> lock(mu);
  of << row.str();
}

std::string visitor_name(const char *suffix, const SearchCfg &cfg) {
  // CSV is comma-delimited -- use ';' inside the descriptor so this field
  // does not fragment into extra columns downstream.
  return std::string("dgai(k=") + std::to_string(cfg.k) + ";L=" + std::to_string(cfg.l) +
        ";beam=" + std::to_string(cfg.beam) + ")_" + suffix;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <data_raw> <queries_raw> <index_prefix> <output_prefix>\n";
    return 1;
  }
  std::string data = argv[1], queries = argv[2], index_prefix = argv[3], output_prefix = argv[4];

  clk::time_point t0 = clk::now();
  auto now_ns = [&] {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
  };

  // Phase 0: bootstrap a small on-disk graph.
  std::string bootstrap_bin = index_prefix + "_bootstrap_data.bin";
  write_bootstrap_bin(data, bootstrap_bin, BOOTSTRAP, DIM);
  std::string build_params = std::to_string(BUILD_R) + " " + std::to_string(BUILD_L) + " " +
                             std::to_string(BUILD_B_GB) + " " + std::to_string(BUILD_M_GB) + " " +
                             std::to_string(BUILD_THREADS);
  bool built = pipeann::build_disk_index<T>(bootstrap_bin.c_str(), index_prefix.c_str(), build_params.c_str(),
                                            pipeann::Metric::L2, false);
  if (!built) {
    std::cerr << "build_disk_index failed\n";
    return 1;
  }

  pipeann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_DISK);
  paras.Set<unsigned>("R_disk", R_DISK);
  paras.Set<float>("alpha_disk", ALPHA_DISK);
  paras.Set<unsigned>("C", C_DISK);
  paras.Set<unsigned>("beamwidth", BEAMWIDTH);
  paras.Set<unsigned>("nodes_to_cache", NODES_TO_CACHE);
  paras.Set<unsigned>("num_threads", NUM_THREADS);

  pipeann::DistanceL2 dist_cmp;
  pipeann::DynamicSSDIndex<T, TagT> index(paras, index_prefix, index_prefix, &dist_cmp, pipeann::Metric::L2,
                                          BEAM_SEARCH, false, 0);

  std::ofstream inserts_of(output_prefix + "inserts.csv");
  std::ofstream queries_of(output_prefix + "queries.csv");
  write_insert_header(inserts_of);
  write_query_header(queries_of);
  std::mutex inserts_mu, queries_mu;

  // Phase 1: plain sequential insert, BOOTSTRAP -> BASE, no search.
  std::vector<T> buf;
  for (size_t off = BOOTSTRAP; off < BASE; off += BATCH) {
    size_t n = std::min(BATCH, BASE - off);
    read_raw(data, off, n, DIM, buf);
    for (size_t i = 0; i < n; i++) {
      TagT tag = (TagT) (off + i);
      uint64_t ios_before = gs->update_ios;
      double io_us_before = gs->insert_io;
      uint64_t start_ns = now_ns();
      auto s = clk::now();
      index.insert(buf.data() + i * DIM, tag);
      uint64_t lat_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - s).count();
      uint64_t n_ios = (gs->update_ios - ios_before) / SECTOR_LEN;
      double io_us = gs->insert_io - io_us_before;
      write_insert_row(inserts_of, inserts_mu, "seq", off / BATCH, tag, start_ns, lat_ns, n_ios, io_us);
    }
    inserts_of.flush();
    std::cerr << "[phase1] inserted up to " << (off + n) << "/" << BASE << "\n";
  }

  // Fixed QLEN-point query prefix, loaded once, reused every phase-2 step.
  std::vector<T> qbuf;
  read_raw(queries, 0, QLEN, DIM, qbuf);

  // Phase 2: windowed run, BASE -> N.
  for (size_t off = BASE; off < N; off += BATCH) {
    size_t n = std::min(BATCH, N - off);
    read_raw(data, off, n, DIM, buf);
    uint64_t batch_id = off / BATCH;

    std::atomic<bool> insert_done{false};
    std::atomic<uint64_t> qi_ctr{0};
    std::atomic<uint64_t> mix_ctr{0};

    std::thread insert_thread([&] {
      for (size_t i = 0; i < n; i++) {
        TagT tag = (TagT) (off + i);
        uint64_t ios_before = gs->update_ios;
        double io_us_before = gs->insert_io;
        uint64_t start_ns = now_ns();
        auto s = clk::now();
        index.insert(buf.data() + i * DIM, tag);
        uint64_t lat_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - s).count();
        uint64_t row = mix_ctr.fetch_add(1);
        if (row % MIXRATE == 0) {
          uint64_t n_ios = (gs->update_ios - ios_before) / SECTOR_LEN;
          double io_us = gs->insert_io - io_us_before;
          write_insert_row(inserts_of, inserts_mu, "mix", batch_id, tag, start_ns, lat_ns, n_ios, io_us);
        }
      }
      insert_done.store(true);
    });

    std::string mix_visitor = visitor_name("mix", PRIMARY_CFG);
    std::vector<std::thread> search_threads;
    for (unsigned t = 0; t < SEARCH_THREADS; t++) {
      search_threads.emplace_back([&] {
        std::vector<TagT> ids(K);
        std::vector<float> dists(K);
        while (!insert_done.load()) {
          size_t qi = qi_ctr.fetch_add(1) % QLEN;
          pipeann::QueryStats stats;
          uint64_t start_ns = now_ns();
          auto s = clk::now();
          index.search(qbuf.data() + qi * DIM, PRIMARY_CFG.k, 0, PRIMARY_CFG.l, PRIMARY_CFG.beam, ids.data(),
                       dists.data(), &stats);
          uint64_t lat_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - s).count();
          uint64_t row = mix_ctr.fetch_add(1);
          if (row % MIXRATE == 0) {
            write_query_row(queries_of, queries_mu, batch_id, mix_visitor, qi, start_ns, lat_ns, ids.data(),
                            dists.data(), stats);
          }
        }
      });
    }
    insert_thread.join();
    for (auto &th : search_threads) th.join();
    inserts_of.flush();
    queries_of.flush();
    std::cerr << "[phase2] mix batch " << batch_id << " done, corpus=" << (off + n) << "\n";

    // Un-undersampled search-only sweep at the just-reached corpus size.
    for (auto &cfg : EXTRA_CFGS) {
      std::string sweep_visitor = visitor_name("run0", cfg);
#pragma omp parallel for num_threads(SEARCH_THREADS) schedule(dynamic)
      for (int64_t qi = 0; qi < (int64_t) QLEN; qi++) {
        std::vector<TagT> ids(K);
        std::vector<float> dists(K);
        pipeann::QueryStats stats;
        uint64_t start_ns = now_ns();
        auto s = clk::now();
        index.search(qbuf.data() + qi * DIM, cfg.k, 0, cfg.l, cfg.beam, ids.data(), dists.data(), &stats);
        uint64_t lat_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - s).count();
        write_query_row(queries_of, queries_mu, batch_id, sweep_visitor, (size_t) qi, start_ns, lat_ns, ids.data(),
                        dists.data(), stats);
      }
      queries_of.flush();
    }
  }

  return 0;
}
