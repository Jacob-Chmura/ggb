#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "common/logging.h"

namespace ggb::bench::perf {

struct BenchStats {
  // Latency (ms)
  double mean, std_dev, min, max;
  double p50, p95, p99;

  // Throughput
  double qps;
  double tps_m;  // Millions of tensors per second
  double gi_bps;

  std::size_t total_queries;
  double total_tensors_m;
};

struct BenchResult {
  std::vector<std::uint64_t> latencies_us;
  std::size_t num_tensors_read{0};
  std::size_t num_elements_per_tensor{0};

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
    const double mean_us = total_us / n;

    auto sq_diff_sum = std::transform_reduce(
        sorted_latencies.begin(), sorted_latencies.end(), 0.0, std::plus<>(),
        [mean_us](auto val) {
          return std::pow(static_cast<double>(val) - mean_us, 2);
        });

    const double std_dev_us = std::sqrt(sq_diff_sum / n);

    auto get_p_ms = [&](double percentile) {
      const size_t idx =
          static_cast<size_t>(std::ceil(percentile / 100.0 * n)) - 1;
      auto val_us = sorted_latencies[std::clamp(idx, std::size_t{0}, n - 1)];
      return static_cast<double>(val_us) / 1000.0;
    };

    const double total_s = total_us / 1'000'000.0;

    return BenchStats{
        .mean = mean_us / 1000.0,
        .std_dev = std_dev_us / 1000.0,
        .min = static_cast<double>(sorted_latencies.front()) / 1000.0,
        .max = static_cast<double>(sorted_latencies.back()) / 1000.0,
        .p50 = get_p_ms(50),
        .p95 = get_p_ms(95),
        .p99 = get_p_ms(99),
        .qps = static_cast<double>(n) / total_s,
        .tps_m = (static_cast<double>(num_tensors_read) / total_s) / 1e6,
        .gi_bps =
            (static_cast<double>(num_tensors_read * num_elements_per_tensor *
                                 sizeof(float))) /
            (total_s * 1024 * 1024 * 1024),
        .total_queries = n,
        .total_tensors_m = static_cast<double>(num_tensors_read) / 1e6};
  }
};

}  // namespace ggb::bench::perf
