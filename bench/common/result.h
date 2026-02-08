#pragma once

#include <cstdint>
#include <vector>

namespace ggb::bench::perf {
struct BenchResult {
  std::vector<std::uint64_t> latencies_us_;
  std::size_t num_tensors_read_{0};
};

}  // namespace ggb::bench::perf
