#pragma once

#include <memory>
#include <utility>

#include "config.h"
#include "ggb.h"
#include "result.h"
#include "timer.h"

namespace ggb::bench {
class Runner {
 public:
  explicit Runner(std::unique_ptr<FeatureStoreBuilder> builder, RunConfig cfg)
      : builder_(std::move(builder)), cfg_(std::move(cfg)) {}

  auto run() -> perf::BenchResult {
    perf::BenchResult result;

    {
      perf::ScopedTimer t("Ingestion");
      ingest_features();
    }

    {
      perf::ScopedTimer t("Building");
      store_ = builder_->build();
    }

    {
      perf::ScopedTimer t("Inference");
    }

    return result;
  }

 private:
  std::unique_ptr<FeatureStoreBuilder> builder_;
  std::unique_ptr<FeatureStore> store_;
  RunConfig cfg_;

  void ingest_features() {}
};
}  // namespace ggb::bench
