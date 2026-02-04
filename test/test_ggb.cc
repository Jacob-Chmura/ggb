#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ggb.h"

constexpr std::string db_path = "ggb.db";
const std::size_t num_nodes = 5;
const std::size_t feature_dim = 3;

using EdgeList = std::vector<std::pair<ggb::NodeID_t, ggb::NodeID_t>>;
using NodeFeatures = std::vector<std::vector<float>>;

namespace {

auto print_matrix(const NodeFeatures& data) -> void {
  for (std::size_t i = 0; i < data.size(); ++i) {
    std::cout << "Node: " << i << " | Feat: [";
    for (auto feat : data[i]) {
      std::cout << feat << ", ";
    }
    std::cout << "]\n";
  }
}

auto generate_graph() -> std::pair<EdgeList, ggb::InMemoryFeatureStore> {
  const EdgeList edges = {{1, 4}, {1, 5}, {2, 3}, {4, 5}};
  NodeFeatures features(num_nodes, std::vector<float>(feature_dim));

  std::random_device rng;
  std::mt19937 engine(rng());
  for (std::size_t i = 0; i < num_nodes; ++i) {
    for (std::size_t j = 0; j < feature_dim; ++j) {
      std::uniform_real_distribution<float> dist(0.0F, 1.0F);
      features[i][j] = dist(engine);
    }
  }
  std::cout << "---------- FULL FEATURES ---------\n";
  ::print_matrix(features);
  return {edges, ggb::InMemoryFeatureStore(features)};
}
}  // namespace

auto main() -> int {
  auto [edges, features] = generate_graph();

  const ggb::Config cfg = {.db_path = db_path};
  auto ctx = ggb::init(cfg);

  ggb::build(ctx, edges, features);

  std::vector<ggb::NodeID_t> nodes = {0, 1, 3};
  auto batch_feats = ggb::gather(ctx, nodes);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    std::cout << "Node: " << nodes[i] << " | Feat: [";
    for (auto feat : batch_feats[i]) {
      std::cout << feat << ", ";
    }
    std::cout << "]\n";
  }
  return 0;
}
