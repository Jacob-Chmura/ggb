#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "common/io.h"
#include "common/logging.h"
#include "config.h"
#include "ggb/core.h"
#include "queries.h"
#include "result.h"
#include "timer.h"

namespace ggb::bench {
class Runner {
 public:
  explicit Runner(std::unique_ptr<FeatureStoreBuilder> builder, RunConfig cfg)
      : builder_(std::move(builder)), cfg_(std::move(cfg)) {}

  auto run() -> perf::BenchResult {
    GGB_LOG_INFO("Starting Benchmark Runner");
    perf::BenchResult result{.cfg = cfg_, .latencies_us_{}};

    {
      const perf::ScopedTimer timer("Ingestion");
      GGB_LOG_INFO("Ingesting features and graph topology");
      ggb::io::ingest_features_from_csv(cfg_.node_feat_path, *builder_);
      ggb::io::ingest_edgelist_from_csv(cfg_.edge_list_path.string(),
                                        edge_buffer_);
      graph_ = ggb::GraphTopology{.edges = std::span(edge_buffer_)};
    }

    {
      const perf::ScopedTimer timer("Building");
      GGB_LOG_INFO("Constructing FeatureStore engine");
      store_ = builder_->build(graph_);
    }

    // Clear edge buffer to free some RAM
    edge_buffer_.clear();
    edge_buffer_.shrink_to_fit();

    {
      GGB_LOG_INFO("Running query workload");
      run_queries(result);
    }

    return result;
  }

 private:
  std::unique_ptr<FeatureStoreBuilder> builder_;
  std::unique_ptr<FeatureStore> store_;
  RunConfig cfg_{};
  std::optional<ggb::GraphTopology> graph_;
  std::vector<std::pair<ggb::NodeID, ggb::NodeID>> edge_buffer_;

  auto run_queries(perf::BenchResult& result) -> void {
    result.num_elements_per_tensor = store_->get_tensor_size().value_or(0);

    const auto queries = QueryLoader::from_csv(cfg_.query_csv_path.string());
    for (const auto& query : queries) {
      {
        const perf::ScopedTimer timer(
            [&](std::uint64_t us) { result.latencies_us_.push_back(us); });
        auto feats = store_->get_multi_tensor(std::span(query));
      }

      result.num_tensors_read_ += query.size();
    }
  }
};
}  // namespace ggb::bench
