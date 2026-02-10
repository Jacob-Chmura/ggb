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
#include "sinks.h"
#include "stats.h"
#include "timer.h"

namespace ggb::bench {
class Runner {
 public:
  explicit Runner(std::unique_ptr<FeatureStoreBuilder> builder, RunConfig cfg)
      : builder_(std::move(builder)), cfg_(std::move(cfg)) {
    add_sink(std::make_unique<LogSink>());
    add_sink(std::make_unique<JsonSink>());
  }

  auto add_sink(std::unique_ptr<ResultSink> sink) -> void {
    sinks_.push_back(std::move(sink));
  }

  auto run() -> void {
    GGB_LOG_INFO("Starting Benchmark Runner");
    BenchResult result;

    {
      const ScopedTimer timer("Ingestion");
      GGB_LOG_INFO("Ingesting features and graph topology");
      ggb::io::ingest_features_from_csv(cfg_.node_feat_path, *builder_);
      ggb::io::ingest_edgelist_from_csv(cfg_.edge_list_path.string(),
                                        edge_buffer_);
      graph_ = ggb::GraphTopology{.edges = std::span(edge_buffer_)};
    }

    {
      const ScopedTimer timer("Building");
      GGB_LOG_INFO("Constructing FeatureStore engine");
      store_ = builder_->build(graph_);
      result.num_elements_per_tensor = store_->get_tensor_size().value_or(0);
    }

    // Clear edge buffer to free some RAM
    edge_buffer_.clear();
    edge_buffer_.shrink_to_fit();

    // Load in queries before taking an IO snapshot
    const auto queries = QueryLoader::from_csv(cfg_.query_csv_path.string());

    const auto* const vmtouch_cmd = "vmtouch -e test.ggb";
    GGB_LOG_INFO("Forcing Kernel Page eviction using vmtouch cmd: ",
                 vmtouch_cmd);
    const auto vmtouch_result = std::system(vmtouch_cmd);
    if (vmtouch_result == 0) {
      GGB_LOG_INFO("Succesfully evicted feature store from OS cache.");
    } else {
      GGB_LOG_WARN(
          "Failed to call vmtouch cmd. Resident pages of the FeatureStore may "
          "still be in memory");
    }

    GGB_LOG_INFO("Running query workload");
    result.on_start();

    for (const auto& query : queries) {
      {
        const ScopedTimer timer(
            [&](std::uint64_t us) { result.record_query(us, query.size()); });
        auto feats = store_->get_multi_tensor(std::span(query));
      }
    }
    result.on_stop();

    auto stats = result.compute_stats();
    for (const auto& sink : sinks_) {
      sink->report(cfg_, stats);
    }
  }

 private:
  std::unique_ptr<FeatureStoreBuilder> builder_;
  std::unique_ptr<FeatureStore> store_;
  std::optional<ggb::GraphTopology> graph_;
  std::vector<std::pair<ggb::NodeID, ggb::NodeID>> edge_buffer_;

  RunConfig cfg_{};
  std::vector<std::unique_ptr<ResultSink>> sinks_;
};

[[nodiscard]] inline auto create_runner(const EngineConfig& engine_type,
                                        RunConfig base_cfg) -> Runner {
  base_cfg.engine = engine_type;
  auto builder = ggb::create_builder(engine_type);
  return Runner(std::move(builder), std::move(base_cfg));
}
}  // namespace ggb::bench
