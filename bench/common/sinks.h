#pragma once
#ifndef GGB_GIT_HASH
#define GGB_GIT_HASH "unknown"  // Fallback
#endif

#include <chrono>
#include <filesystem>
#include <format>
#include <iomanip>
#include <sstream>
#include <string>
#include <variant>

#include "common/config.h"
#include "common/logging.h"
#include "common/stats.h"
#include "ggb/core.h"

namespace ggb::bench {

class ResultSink {
 public:
  virtual ~ResultSink() = default;
  virtual auto report(const RunConfig& cfg, const BenchStats& stats)
      -> void = 0;
};

class LogSink : public ResultSink {
 public:
  auto report(const RunConfig& cfg, const BenchStats& stats) -> void override {
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
        << std::format(" {:<20} : {:>12.2f} MM\n", "Total Tensors",
                       static_cast<double>(stats.total_tensors) / 1e6)
        << std::string(60, '-')
        << "\n"
        // Throughput
        << std::format(" {:<20} : {:>12.2f} req/s\n", "Throughput QPS",
                       stats.qps)
        << std::format(" {:<20} : {:>12.3f} MM/s\n", "Throughput TPS",
                       stats.tps_m)
        << std::format(" {:<20} : {:>12.2f} GB/s\n", "Throughput BW",
                       stats.gi_bps)
        << std::string(60, '-')
        << "\n"
        // System IO
        << std::format(" {:<20} : {:>12.3f} GB\n", "Peak RAM",
                       stats.peak_ram_gb)
        << std::format(" {:<20} : {:>12.3f} GB\n", "Disk Read",
                       stats.disk_read_gb)
        << std::format(" {:<20} : {:>12.2f} GB/s\n", "Disk IOPS",
                       stats.disk_iops_gb)
        << std::format(" {:<20} : {:>12} hits\n", "Major Faults",
                       stats.major_faults)
        << std::format(" {:<20} : {:>12} hits\n", "Minor Faults",
                       stats.minor_faults)
        << std::string(60, '-')
        << "\n"
        // Scheduler context switches (indicate IO blocking)
        << std::format(" {:<20} : {:>12} \n", "CS (Voluntary)",
                       stats.vol_context_switches)
        << std::format(" {:<20} : {:>12} \n", "CS (Involuntary)",
                       stats.invol_context_switches)
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

class JsonSink : public ResultSink {
 public:
  auto report(const RunConfig& cfg, const BenchStats& stats) -> void override {
    auto results_dir = cfg.get_results_dir();
    std::filesystem::create_directories(results_dir);

    std::string engine_name = std::visit(
        overloaded{[](const FlatMmapConfig&) { return "mmap"; },
                   [](const InMemoryConfig&) { return "in_memory"; }},
        cfg.engine);

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d_%H-%M-%S");

    std::string filename =
        std::format("result_{}_{}.json", engine_name, ss.str());
    auto file_path = results_dir / filename;

    nlohmann::json out;
    out["metadata"] = {{"dataset", cfg.dataset_name},
                       {"run_id", cfg.run_id},
                       {"engine", engine_name},
                       {"git_hash", GGB_GIT_HASH},
                       {"sampling", cfg.sampling}};
    out["stats"] = stats;

    std::ofstream f(file_path);
    f << out.dump(4);  // Indent 4 spaces
    GGB_LOG_INFO("Results saved to: {}", file_path.string());
  }

 private:
  template <class... Ts>
  struct overloaded : Ts... {
    using Ts::operator()...;
  };
};

}  // namespace ggb::bench
