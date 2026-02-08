#pragma once
#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."  // Fallback
#endif

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace ggb::bench {

constexpr std::string node_feat_file_name = "node-feat.csv";
constexpr std::string edge_list_file_name = "edge.csv";

struct RunConfig {
  struct SamplingConfig {
    int seed;
    std::size_t batch_size;
    std::size_t num_hops;
    std::size_t fan_out;
  };

  std::string dataset_name;
  std::string run_id;
  std::filesystem::path node_feat_path;
  std::filesystem::path edge_list_path;

  SamplingConfig sampling_config;

  std::vector<std::filesystem::path> query_csvs;

  static auto load_from_run(const std::string& dataset_name,
                            const std::string& run_id)
      -> std::optional<RunConfig> {
    const auto dataset_path = get_dataset_path(dataset_name);

    std::error_code ec;
    if (!std::filesystem::exists(dataset_path, ec)) {
      std::cerr << std::format("Error: Dataset path not found: {}\n",
                               dataset_path.string());
      return std::nullopt;
    }

    RunConfig cfg;
    cfg.dataset_name = dataset_name;
    cfg.run_id = run_id;
    cfg.node_feat_path = dataset_path / node_feat_file_name;
    cfg.edge_list_path = dataset_path / edge_list_file_name;

    if (!std::filesystem::exists(cfg.node_feat_path, ec)) {
      std::cerr << "Error: Node feature file does not exists: "
                << cfg.node_feat_path << "\n";
      return std::nullopt;
    }

    if (!std::filesystem::exists(cfg.edge_list_path, ec)) {
      std::cerr << "Error: Edge list file does not exists: "
                << cfg.edge_list_path << "\n";
      return std::nullopt;
    }

    // TODO(kuba): parse metadata json
    cfg.sampling_config.seed = 1337;
    cfg.sampling_config.batch_size = 1024;
    cfg.sampling_config.num_hops = 2;
    cfg.sampling_config.fan_out = 10;

    auto run_dir = dataset_path / "queries" / run_id;
    if (!std::filesystem::exists(run_dir, ec)) {
      std::cerr << "Error: Run ID directory does not exist: " << run_dir
                << "\n";
      return std::nullopt;
    }

    for (const auto& entry : std::filesystem::directory_iterator(run_dir)) {
      if (entry.path().extension() == ".csv") {
        cfg.query_csvs.push_back(entry.path());
      }
    }
    if (cfg.query_csvs.empty()) {
      std::cerr << "Error: No query CSVs found in " << run_dir << "\n";
      return std::nullopt;
    }
    std::ranges::sort(cfg.query_csvs);  // Sort so queries run in seeded order
    return cfg;
  }

 private:
  [[nodiscard]] static auto get_dataset_path(const std::string& dataset_name)
      -> std::filesystem::path {
    return std::filesystem::path(PROJECT_ROOT) / "bench" / "data" /
           dataset_name;
  }
};

}  // namespace ggb::bench
