#pragma once

#include <format>
#include <sstream>
#include <string>

#include "common/config.h"
#include "common/logging.h"
#include "common/stats.h"
#include "ggb/core.h"

namespace ggb::bench::perf {

class ResultSink {
 public:
  virtual ~ResultSink() = default;
  virtual void report(const RunConfig& cfg, const BenchStats& stats) = 0;
};

class LogSink : public ResultSink {
 public:
  void report(const RunConfig& cfg, const BenchStats& stats) override {
    std::string engine_info = std::visit(
        overloaded{
            [](const FlatMmapConfig& c) {
              return std::format("FlatMmap (path: {})", c.db_path);
            },
            [](const InMemoryConfig&) { return std::string("InMemory"); }},
        cfg.engine);

    std::ostringstream oss;
    const auto sampling_str =
        std::format("batch={}, hops={}, fanout={}", cfg.sampling.batch_size,
                    cfg.sampling.num_hops, cfg.sampling.fan_out);
    oss << "\n"
        << std::string(60, '=') << "\n"
        << std::format(" {:^58} \n", "BENCHMARK: " + cfg.dataset_name)
        << std::string(60, '=')
        << "\n"
        // Metadata
        << std::format(" {:<20} : {}\n", "Run ID", cfg.run_id)
        << std::format(" {:<20} : {}\n", "Engine Type", engine_info)
        << std::format(" {:<20} : {}\n", "Sampling", sampling_str)
        << std::string(60, '-')
        << "\n"
        // Counters
        << std::format(" {:<20} : {:>12} reqs\n", "Total Queries",
                       stats.total_queries)
        << std::format(" {:<20} : {:>12.3f} MM\n", "Total Tensors",
                       stats.total_tensors_m)
        << std::string(60, '-')
        << "\n"
        // Throughput
        << std::format(" {:<20} : {:>12.2f} req/s\n", "Throughput QPS",
                       stats.qps)
        << std::format(" {:<20} : {:>12.3f} MM/s\n", "Throughput TPS",
                       stats.tps_m)
        << std::format(" {:<20} : {:>12.2f} GiB/s\n", "Throughput BW",
                       stats.gi_bps)
        << std::string(60, '-')
        << "\n"
        // Latency
        << std::format(" {:<20} : {:>12.3f} ms\n", "Latency Mean", stats.mean)
        << std::format(" {:<20} : {:>12.3f} ms\n", "Latency StdDev",
                       stats.std_dev)
        << std::format(" {:<20} : {:>12.3f} ms\n", "Latency P50", stats.p50)
        << std::format(" {:<20} : {:>12.3f} ms\n", "Latency P99", stats.p99)
        << std::format(" {:<20} : {:>12.3f} ms\n", "Latency Max", stats.max)
        << std::string(60, '=');

    GGB_LOG_INFO("{}", oss.str());
  }

 private:
  template <class... Ts>
  struct overloaded : Ts... {
    using Ts::operator()...;
  };
};
}  // namespace ggb::bench::perf
