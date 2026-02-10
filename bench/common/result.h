#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "common/logging.h"
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
  std::size_t num_elements_per_tensor{0};

  auto print() -> void {
    GGB_LOG_INFO(
        "Results for {}, run_id: {}, batch_size: {}, num_hops: {}, fan_out: "
        "{}",
        cfg.dataset_name, cfg.run_id, cfg.sampling.batch_size,
        cfg.sampling.num_hops, cfg.sampling.fan_out);

    if (latencies_us_.empty()) {
      GGB_LOG_WARN("No latencies found");
      return;
    }

    std::ranges::sort(latencies_us_);

    const auto n = latencies_us_.size();
    const double total_latency =
        std::accumulate(latencies_us_.begin(), latencies_us_.end(), 0.0);
    const double mean = total_latency / n;

    auto sq_diff_sum = std::transform_reduce(
        latencies_us_.begin(), latencies_us_.end(), 0.0, std::plus<>(),
        [mean](auto val) { return std::pow(val - mean, 2); });
    const double std_dev = std::sqrt(sq_diff_sum / n);

    auto get_p_ms = [&](double percentile) {
      const size_t idx =
          static_cast<size_t>(std::ceil(percentile / 100.0 * n)) - 1;
      auto val_us = latencies_us_[std::clamp(idx, std::size_t{0}, n - 1)];
      return val_us / 1000.0;  // Convert to ms
    };

    const auto total_latency_s = total_latency / 1'000'000.0;
    const auto qps = static_cast<double>(n) / total_latency_s;
    const auto tps = static_cast<double>(num_tensors_read_) / total_latency_s;

    const double total_tensors_m = static_cast<double>(num_tensors_read_) / 1e6;
    const double tps_m = tps / 1e6;

    const double gb_ps =
        (static_cast<double>(num_tensors_read_ * num_elements_per_tensor *
                             sizeof(float))) /
        (total_latency_s * 1024 * 1024 * 1024);

    std::ostringstream oss;
    oss << "\n"
        << std::string(60, '=') << "\n"
        << std::format(" {:^58} \n", "BENCHMARK: " + cfg.dataset_name)
        << std::string(60, '=') << "\n"
        << std::format(" {:<20} : {:>10}\n", "Total Queries", n)
        << std::format(" {:<20} : {:>10.3f} MM\n", "Total Tensors",
                       total_tensors_m)
        << std::string(60, '-') << "\n"
        << std::format(" {:<20} : {:>10.2f} req/s\n", "Queries/sec", qps)
        << std::format(" {:<20} : {:>10.3f} MM/s\n", "Tensors/sec", tps_m)
        << std::format(" {:<20} : {:>10.2f} GiB/s\n", "Throughput", gb_ps)
        << std::string(60, '-') << "\n"
        << std::format(" {:<20} : {:>10.3f} ms\n", "Latency Mean",
                       mean / 1000.0)
        << std::format(" {:<20} : {:>10.3f} ms\n", "Latency StdDev",
                       std_dev / 1000.0)
        << std::format(" {:<20} : {:>10.3f} ms\n", "Latency P50", get_p_ms(50))
        << std::format(" {:<20} : {:>10.3f} ms\n", "Latency P95", get_p_ms(95))
        << std::format(" {:<20} : {:>10.3f} ms\n", "Latency P99", get_p_ms(99))
        << std::string(60, '=');

    GGB_LOG_INFO("{}", oss.str());
  }
};

}  // namespace ggb::bench::perf
