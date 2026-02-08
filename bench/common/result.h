#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "config.h"

namespace ggb::bench::perf {

// TODO(kuba): Create isolated stats obj
struct LatencyStats {
  double mean, std_dev, min, max;
  double p50, p90, p95, p99;
};

// TODO(kuba): Create sinks
struct BenchResult {
  RunConfig cfg;
  std::vector<std::uint64_t> latencies_us_;
  std::size_t num_tensors_read_{0};

  auto print() -> void {
    std::cout << std::format(
        "Results for {}, run_id: {}, batch_size: {}, num_hops: {}, fan_out: "
        "{}\n",
        cfg.dataset_name, cfg.run_id, cfg.sampling.batch_size,
        cfg.sampling.num_hops, cfg.sampling.fan_out);

    if (latencies_us_.empty()) {
      std::cerr << "Error: No latencies found\n";
      return;
    }

    std::ranges::sort(latencies_us_);

    const size_t n = latencies_us_.size();
    const double sum =
        std::accumulate(latencies_us_.begin(), latencies_us_.end(), 0.0);
    const double mean = sum / n;

    auto sq_diff_sum = std::transform_reduce(
        latencies_us_.begin(), latencies_us_.end(), 0.0, std::plus<>(),
        [mean](auto val) { return std::pow(val - mean, 2); });
    const double std_dev = std::sqrt(sq_diff_sum / n);

    auto get_p_ms = [&](double percentile) {
      size_t idx = static_cast<size_t>(std::ceil(percentile / 100.0 * n)) - 1;
      auto val_us = latencies_us_[std::clamp(idx, std::size_t(0), n - 1)];
      return val_us / 1000.0;  // Convert to ms
    };

    std::cout << "----------------------------------------------------------\n";
    std::cout << std::format("{:<15} : {}\n", "Total Queries", n);
    std::cout << std::format("{:<15} : {}\n", "Total Tensors",
                             num_tensors_read_);
    std::cout << std::format("{:<15} : {:.3f} ms\n", "Mean Latency",
                             mean / 1000.0);
    std::cout << std::format("{:<15} : {:.3f} ms\n", "Std Deviation",
                             std_dev / 1000.0);
    std::cout << "----------------------------------------------------------\n";
    std::cout << std::format(
        "{:<15} : {:.3f} ms\n", "Min",
        static_cast<double>(latencies_us_.front()) / 1000.0);
    std::cout << std::format("{:<15} : {:.3f} ms\n", "P50", get_p_ms(50));
    std::cout << std::format("{:<15} : {:.3f} ms\n", "P90", get_p_ms(90));
    std::cout << std::format("{:<15} : {:.3f} ms\n", "P95", get_p_ms(95));
    std::cout << std::format("{:<15} : {:.3f} ms\n", "P99", get_p_ms(99));
    std::cout << std::format(
        "{:<15} : {:.3f} ms\n", "Max",
        static_cast<double>(latencies_us_.back()) / 1000.0);
    std::cout << "----------------------------------------------------------\n";
  }
};

}  // namespace ggb::bench::perf
