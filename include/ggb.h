#pragma once

#include <fstream>
#include <iostream>
#include <span>
#include <vector>

namespace ggb {

using NodeID_t = unsigned long long;

struct ggb_config {
  std::string db_path;
};

struct ggb_context {
  ggb_config config;
};

class FeatureStore {
 public:
  virtual ~FeatureStore() = default;

  virtual auto num_nodes() const -> size_t = 0;
  virtual auto feature_dim() const -> size_t = 0;
  virtual auto get_feature(NodeID_t node) const -> std::span<const float> = 0;
};

class InMemoryFeatureStore final : public FeatureStore {
 public:
  explicit InMemoryFeatureStore(std::vector<std::vector<float>> data)
      : _data(std::move(data)),
        _num_nodes(_data.size()),
        _feature_dim(_data.empty() ? 0 : _data[0].size()) {}

  size_t num_nodes() const override { return _num_nodes; }
  size_t feature_dim() const override { return _feature_dim; }

  std::span<const float> get_feature(NodeID_t node) const override {
    return std::span<const float>(_data[node]);
  }

 private:
  std::vector<std::vector<float>> _data;
  size_t _num_nodes = 0;
  size_t _feature_dim = 0;
};

namespace detail {
auto write_file(const std::string &db_path, const FeatureStore &features)
    -> void {
  std::cout << "Building graph at db_path: " << db_path << std::endl;
  std::ofstream out_file(db_path, std::ios::binary);
  if (!out_file) {
    std::cerr << "Error opening file: " << db_path << std::endl;
    return;
  }

  const auto N = features.num_nodes();
  const auto F = features.feature_dim();
  if (N == 0 || F == 0) {
    std::cerr << "Got empty node features" << std::endl;
    return;
  }

  out_file.write(reinterpret_cast<const char *>(&N), sizeof(N));
  out_file.write(reinterpret_cast<const char *>(&F), sizeof(F));
  for (size_t node = 0; node < N; ++node) {
    const auto x = features.get_feature(node);
    if (x.size() != F) {
      std::cerr << "Row has wrong feature size" << std::endl;
      return;
    }
    out_file.write(reinterpret_cast<const char *>(x.data()), F * sizeof(x[0]));
  }
}

auto read_file(const std::string &db_path) -> InMemoryFeatureStore {
  std::cout << "Reading graph at db_path: " << db_path << std::endl;
  std::ifstream in_file(db_path, std::ios::binary);
  if (!in_file) {
    std::cerr << "Error opening file: " << db_path << std::endl;
    return InMemoryFeatureStore({});
  }

  size_t N = 0;
  size_t F = 0;
  in_file.read(reinterpret_cast<char *>(&N), sizeof(N));
  in_file.read(reinterpret_cast<char *>(&F), sizeof(F));

  if (N == 0 || F == 0) {
    std::cerr << "Empty matrix in db" << std::endl;
    return InMemoryFeatureStore({});
  }

  std::vector<std::vector<float>> features(N, std::vector<float>(F));
  for (size_t i = 0; i < N; ++i) {
    in_file.read(reinterpret_cast<char *>(features[i].data()),
                 F * sizeof(float));
  }
  return InMemoryFeatureStore(features);
}

}  // namespace detail

auto init(const ggb_config &config) -> ggb_context {
  ggb_context ctx = {.config = config};
  return ctx;
}

auto build(const ggb_context ctx,
           std::span<const std::pair<NodeID_t, NodeID_t>> edges,
           const FeatureStore &features) -> void {
  std::cout << "Number of edges: " << edges.size() << std::endl;
  std::cout << "Number of nodes: " << features.num_nodes() << std::endl;
  std::cout << "Feature Dimension: " << features.feature_dim() << std::endl;
  detail::write_file(ctx.config.db_path, features);
}

auto gather(const ggb_context ctx, const std::vector<NodeID_t> &nodes)
    -> std::vector<std::vector<float>> {
  const auto features = detail::read_file(ctx.config.db_path);
  const size_t B = nodes.size();
  const size_t F = features.feature_dim();

  std::vector<std::vector<float>> batch_features(B, std::vector<float>(F));
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto x = features.get_feature(nodes[i]);
    batch_features[i].assign(x.begin(), x.end());
  }
  return batch_features;
}

}  // namespace ggb
