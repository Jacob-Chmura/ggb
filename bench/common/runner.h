#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "ggb.h"
#include "queries.h"
#include "result.h"

namespace ggb::bench {
class Runner {
 public:
  explicit Runner(std::unique_ptr<FeatureStore> store)
      : store_(std::move(store)) {}

  auto run(const std::vector<Query>& queries) -> perf::BenchResult {
    return perf::BenchResult{};
  }

 private:
  std::unique_ptr<FeatureStore> store_;
};
}  // namespace ggb::bench
