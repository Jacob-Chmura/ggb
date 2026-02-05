#ifndef GGB_H
#define GGB_H

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ggb {

using NodeID_t = std::uint64_t;

struct Config {
  std::string db_path;
};

struct Context {
  Config config;
};

class FeatureStore {
 public:
  virtual ~FeatureStore() = default;

  [[nodiscard]] virtual auto num_nodes() const -> std::size_t = 0;
  [[nodiscard]] virtual auto feature_dim() const -> std::size_t = 0;
  [[nodiscard]] virtual auto get_feature(NodeID_t node) const
      -> std::span<const float> = 0;
};

class InMemoryFeatureStore final : public FeatureStore {
 public:
  explicit InMemoryFeatureStore(std::vector<std::vector<float>> data)
      : data_(std::move(data)),
        num_nodes_(data_.size()),
        feature_dim_(data_.empty() ? 0 : data_[0].size()) {}

  [[nodiscard]] auto num_nodes() const -> std::size_t override {
    return num_nodes_;
  }
  [[nodiscard]] auto feature_dim() const -> std::size_t override {
    return feature_dim_;
  }

  [[nodiscard]] auto get_feature(NodeID_t node) const
      -> std::span<const float> override {
    return {data_[node]};
  }

 private:
  std::vector<std::vector<float>> data_;
  std::size_t num_nodes_ = 0;
  std::size_t feature_dim_ = 0;
};

namespace detail {
inline auto write_file(const std::string &db_path, const FeatureStore &features)
    -> void {
  std::cout << "Building graph at db_path: " << db_path << std::endl;
  std::ofstream out_file(db_path, std::ios::binary);
  if (!out_file) {
    std::cerr << "Error opening file: " << db_path << std::endl;
    return;
  }

  const auto num_nodes = features.num_nodes();
  const auto feature_dim = features.feature_dim();
  if (num_nodes == 0 || feature_dim == 0) {
    std::cerr << "Got empty node features\n";
    return;
  }

  out_file.write(reinterpret_cast<const char *>(&num_nodes), sizeof(num_nodes));
  out_file.write(reinterpret_cast<const char *>(&feature_dim),
                 sizeof(feature_dim));
  for (std::size_t node = 0; node < num_nodes; ++node) {
    const auto feat = features.get_feature(node);
    if (feat.size() != feature_dim) {
      std::cerr << "Row has wrong feature size\n";
      return;
    }
    out_file.write(reinterpret_cast<const char *>(feat.data()),
                   feature_dim * sizeof(feat[0]));
  }
}

inline auto read_file(const std::string &db_path) -> InMemoryFeatureStore {
  std::cout << "Reading graph at db_path: " << db_path << std::endl;
  std::ifstream in_file(db_path, std::ios::binary);
  if (!in_file) {
    std::cerr << "Error opening file: " << db_path << std::endl;
    return InMemoryFeatureStore({});
  }

  std::size_t num_nodes = 0;
  std::size_t feature_dim = 0;
  in_file.read(reinterpret_cast<char *>(&num_nodes), sizeof(num_nodes));
  in_file.read(reinterpret_cast<char *>(&feature_dim), sizeof(feature_dim));

  if (num_nodes == 0 || feature_dim == 0) {
    std::cerr << "Got empty node features\n";
    return InMemoryFeatureStore({});
  }

  std::vector<std::vector<float>> features(num_nodes,
                                           std::vector<float>(feature_dim));
  for (std::size_t i = 0; i < num_nodes; ++i) {
    in_file.read(reinterpret_cast<char *>(features[i].data()),
                 feature_dim * sizeof(float));
  }
  return InMemoryFeatureStore(features);
}

}  // namespace detail

inline auto init(const Config &config) -> Context {
  Context ctx = {.config = config};
  return ctx;
}

inline auto build(const Context &ctx,
                  std::span<const std::pair<NodeID_t, NodeID_t>> edges,
                  const FeatureStore &features) -> void {
  std::cout << "Number of edges: " << edges.size() << std::endl;
  std::cout << "Number of nodes: " << features.num_nodes() << std::endl;
  std::cout << "Feature Dim: " << features.feature_dim() << std::endl;
  detail::write_file(ctx.config.db_path, features);
}

inline auto gather(const Context &ctx, const std::vector<NodeID_t> &nodes)
    -> std::vector<std::vector<float>> {
  const auto features = detail::read_file(ctx.config.db_path);
  const std::size_t batch_size = nodes.size();
  const std::size_t feature_dim = features.feature_dim();

  std::vector<std::vector<float>> batch_features(
      batch_size, std::vector<float>(feature_dim));
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto feat = features.get_feature(nodes[i]);
    batch_features[i].assign(feat.begin(), feat.end());
  }
  return batch_features;
}

}  // namespace ggb

#endif  // GGB_H
