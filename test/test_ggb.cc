#include <random>

#include "ggb.h"

const std::string DB_PATH = "ggb.db";
const size_t N = 5;
const size_t F = 3;

using EdgeList = std::vector<std::pair<ggb::NodeID_t, ggb::NodeID_t>>;
using NodeFeatures = std::vector<std::vector<float>>;

auto print_matrix(NodeFeatures& data) -> void {
  for (size_t i = 0; i < data.size(); ++i) {
    std::cout << "Node: " << i << " | Feat: [";
    for (auto x : data[i]) {
      std::cout << x << ", ";
    }
    std::cout << "]\n";
  }
}
auto generate_graph() -> std::pair<EdgeList, ggb::InMemoryFeatureStore> {
  EdgeList edges = {{1, 4}, {1, 5}, {2, 3}, {4, 5}};
  NodeFeatures features(N, std::vector<float>(F));

  std::random_device rd;
  std::mt19937 engine(rd());
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < F; ++j) {
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      features[i][j] = dist(engine);
    }
  }
  std::cout << "---------- FULL FEATURES ---------\n";
  print_matrix(features);
  return {edges, ggb::InMemoryFeatureStore(features)};
}

int main() {
  auto [edges, features] = generate_graph();

  ggb::ggb_config cfg = {.db_path = DB_PATH};
  auto ctx = ggb::init(cfg);

  ggb::build(ctx, edges, features);

  std::vector<ggb::NodeID_t> nodes = {0, 1, 3};
  auto batch_feats = ggb::gather(ctx, nodes);
  for (size_t i = 0; i < nodes.size(); ++i) {
    std::cout << "Node: " << nodes[i] << " | Feat: [";
    for (auto x : batch_feats[i]) {
      std::cout << x << ", ";
    }
    std::cout << "]\n";
  }
  return 0;
}
