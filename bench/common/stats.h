#pragma once

#include <sys/resource.h>
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "common/logging.h"

// Third-party
#include <nlohmann/json.hpp>

namespace ggb::bench {

struct BenchStats {
  // Latency (ms)
  double mean, std_dev, min, max;
  double p50, p95, p99;

  // Throughput
  double qps;
  double tps_m;  // Millions of tensors per second
  double gi_bps;

  // IO & System Metrics
  double peak_ram_gb;
  double disk_read_gb;
  double disk_iops_gb;
  std::uint64_t major_faults;
  std::uint64_t minor_faults;
  std::uint64_t vol_context_switches;
  std::uint64_t invol_context_switches;

  std::size_t total_queries;
  std::size_t total_tensors;
};

inline auto to_json(nlohmann::json& j, const BenchStats& s) -> void {
  j = nlohmann::json{{"mean_latency_ms", s.mean},
                     {"std_dev_latency_ms", s.std_dev},
                     {"min_latency_ms", s.min},
                     {"max_latency_ms", s.max},
                     {"p50_latency_ms", s.p50},
                     {"p95_latency_ms", s.p95},
                     {"p99_latency_ms", s.p99},
                     {"qps_throughput", s.qps},
                     {"tps_mm_throughput", s.tps_m},
                     {"gi_bps_throughput", s.gi_bps},
                     {"peak_ram_gb", s.peak_ram_gb},
                     {"disk_read_gb", s.disk_read_gb},
                     {"disk_iops_gb", s.disk_iops_gb},
                     {"major_faults", s.major_faults},
                     {"minor_faults", s.minor_faults},
                     {"voluntary_context_switches", s.vol_context_switches},
                     {"involuntary_context_switches", s.invol_context_switches},
                     {"total_queries", s.total_queries},
                     {"total_tensors", s.total_tensors}};
}

struct IOSnapshot {
  std::uint64_t major_faults{0};
  std::uint64_t minor_faults{0};
  std::uint64_t vol_csw{0};
  std::uint64_t invol_csw{0};
  std::uint64_t read_bytes{0};
  double peak_rss_gb{0};

  static auto capture() -> IOSnapshot {
    IOSnapshot snap;
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      snap.major_faults = usage.ru_majflt;
      snap.minor_faults = usage.ru_minflt;
      snap.vol_csw = usage.ru_nvcsw;
      snap.invol_csw = usage.ru_nivcsw;
      // Linux ru_maxrss is in KiB
      snap.peak_rss_gb =
          static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
    } else {
      GGB_LOG_WARN("getrusage failed: {}, IO metrics will be meaningless",
                   errno);
    }

#ifdef __linux__
    std::ifstream io_file("/proc/self/io");
    if (!io_file.is_open()) {
      GGB_LOG_WARN(
          "Cannot open /proc/self/io. Are TASK_IO_ACCOUNTING configs enabled "
          "in kernel?");
      return snap;
    }

    std::string label;
    while (io_file >> label) {
      if (label == "read_bytes:") {
        io_file >> snap.read_bytes;
        return snap;
      }
    }
    GGB_LOG_WARN("'read_bytes:' key not found in /proc/self/io");
#else
    GGB_LOG_INFO("Non-Linux platform detected; skipping /proc/self/io stats.");
#endif
    return snap;
  }
};

struct BenchResult {
  std::vector<std::uint64_t> latencies_us;
  std::size_t num_tensors_read{0};
  std::size_t num_elements_per_tensor{0};

  IOSnapshot start_io;
  IOSnapshot end_io;

  auto on_start() -> void { start_io = IOSnapshot::capture(); }
  auto on_stop() -> void { end_io = IOSnapshot::capture(); }

  auto record_query(std::uint64_t duration_us, std::size_t batch_size) -> void {
    latencies_us.push_back(duration_us);
    num_tensors_read += batch_size;
  }

  [[nodiscard]] auto compute_stats() const -> BenchStats {
    if (latencies_us.empty()) {
      GGB_LOG_WARN("No latencies found");
      return {};
    }

    auto sorted_latencies = latencies_us;
    std::ranges::sort(sorted_latencies);

    const auto n = sorted_latencies.size();
    const double total_us =
        std::accumulate(sorted_latencies.begin(), sorted_latencies.end(), 0.0);
    const double total_s = total_us / 1'000'000.0;
    const double mean_us = total_us / n;

    auto get_p_ms = [&](double percentile) {
      const size_t idx =
          static_cast<size_t>(std::ceil(percentile / 100.0 * n)) - 1;
      return static_cast<double>(
                 sorted_latencies[std::clamp(idx, std::size_t{0}, n - 1)]) /
             1000.0;
    };

    std::uint64_t delta_faults = end_io.major_faults - start_io.major_faults;
    std::uint64_t delta_bytes = end_io.read_bytes - start_io.read_bytes;

    return BenchStats{
        .mean = mean_us / 1000.0,
        .std_dev = std::sqrt(std::transform_reduce(
                                 sorted_latencies.begin(),
                                 sorted_latencies.end(), 0.0, std::plus<>(),
                                 [mean_us](auto v) {
                                   return std::pow(
                                       static_cast<double>(v) - mean_us, 2);
                                 }) /
                             n) /
                   1000.0,
        .min = static_cast<double>(sorted_latencies.front()) / 1000.0,
        .max = static_cast<double>(sorted_latencies.back()) / 1000.0,
        .p50 = get_p_ms(50.0),
        .p95 = get_p_ms(95.0),
        .p99 = get_p_ms(99.0),
        .qps = static_cast<double>(n) / total_s,
        .tps_m = (static_cast<double>(num_tensors_read) / total_s) / 1e6,
        .gi_bps =
            (static_cast<double>(num_tensors_read * num_elements_per_tensor *
                                 sizeof(float))) /
            (total_s * 1024 * 1024 * 1024),
        .peak_ram_gb = end_io.peak_rss_gb,
        .disk_read_gb =
            static_cast<double>(delta_bytes) / (1024.0 * 1024.0 * 1024.0),
        .disk_iops_gb = static_cast<double>(delta_bytes) /
                        (1024.0 * 1024.0 * 1024.0 * total_s),
        .major_faults = delta_faults,
        .minor_faults = end_io.minor_faults - start_io.minor_faults,
        .vol_context_switches = end_io.vol_csw - start_io.vol_csw,
        .invol_context_switches = end_io.invol_csw - start_io.invol_csw,
        .total_queries = n,
        .total_tensors = num_tensors_read};
  }
};

}  // namespace ggb::bench
