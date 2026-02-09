#pragma once
#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."  // Fallback
#endif

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#include "common/logging.h"

namespace ggb::bench {
namespace fs = std::filesystem;

// TODO(kuba): the engines themselves have configs
struct RunConfig {
  struct SamplingParams {
    int seed{0};
    std::size_t batch_size{0};
    std::size_t num_hops{0};
    std::size_t fan_out{0};
  };

  std::string dataset_name;
  std::string run_id;

  fs::path node_feat_path;
  fs::path edge_list_path;
  fs::path query_csv_path;
  SamplingParams sampling;

  [[nodiscard]] static auto load(const std::string_view dataset_name,
                                 const std::string_view run_id)
      -> std::optional<RunConfig> {
    GGB_LOG_INFO("Trying to load Config with dataset: {}, run_id: {}",
                 dataset_name, run_id);

    const auto dataset_dir = get_dataset_dir(dataset_name);
    if (!fs::is_directory(dataset_dir)) {
      GGB_LOG_ERROR("Dataset directory not found: {}", dataset_dir.string());
      return std::nullopt;
    }

    RunConfig cfg;
    cfg.dataset_name = dataset_name;
    cfg.run_id = run_id;
    cfg.node_feat_path = dataset_dir / node_feat_file_name;
    cfg.edge_list_path = dataset_dir / edge_list_file_name;

    if (!fs::exists(cfg.node_feat_path)) {
      GGB_LOG_ERROR("Feature file not found: {}", cfg.node_feat_path.string());
      return std::nullopt;
    }
    if (!fs::exists(cfg.edge_list_path)) {
      GGB_LOG_ERROR("Edgelist file not found: {}", cfg.edge_list_path.string());
      return std::nullopt;
    }

    const auto run_dir = dataset_dir / run_id;
    if (!fs::is_directory(run_dir)) {
      std::cerr << "Error: Run directory does not exist: " << run_dir << "\n";
      GGB_LOG_ERROR("Run directory: {} is not a valid directory",
                    run_dir.string());
      return std::nullopt;
    }

    for (const auto& entry : fs::directory_iterator(run_dir)) {
      if (entry.path().extension() == ".csv") {
        if (!cfg.query_csv_path.empty()) {
          GGB_LOG_WARN("Multiple CSVs found in {}. Skipping: {} (using: {})",
                       run_dir.string(), entry.path().string(),
                       cfg.query_csv_path.string());
          continue;
        }
        cfg.query_csv_path = entry.path();
      }
    }

    if (cfg.query_csv_path.empty()) {
      GGB_LOG_ERROR("No query CSVs found in run directory: {}",
                    run_dir.string());
      return std::nullopt;
    }
    GGB_LOG_INFO("Using query file: {}",
                 cfg.query_csv_path.filename().string());

    // TODO(kuba): json parsing
    cfg.sampling = {
        .seed = 1337, .batch_size = 1024, .num_hops = 2, .fan_out = 10};
    GGB_LOG_WARN("JSON parsing not implemented, using dummy values for now");
    return cfg;
  }

 private:
  [[nodiscard]] static auto get_dataset_dir(const std::string_view dataset_name)
      -> fs::path {
    return std::filesystem::path(PROJECT_ROOT) / "bench" / "data" /
           dataset_name;
  }

  constexpr static std::string node_feat_file_name = "node-feat.csv";
  constexpr static std::string edge_list_file_name = "edge.csv";
};

}  // namespace ggb::bench
