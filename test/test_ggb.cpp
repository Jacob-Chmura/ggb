#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ggb.h"

constexpr std::string db_path = "test.ggb";
const std::size_t num_nodes = 5;
const std::size_t feature_dim = 3;

namespace {
void print_feature_row(ggb::Key key, const std::optional<ggb::Value>& feat) {
  std::cout << "Node: " << key << " | Feat: ";
  if (!feat.has_value()) {
    std::cout << "[Not Found]\n";
    return;
  }
  std::cout << "[";
  for (auto val : feat.value()) {
    std::cout << val << (val == feat->back() ? "" : ", ");
  }
  std::cout << "]\n";
}

void ingest_data(ggb::FeatureStoreBuilder& builder) {
  std::random_device rng;
  std::mt19937 engine(rng());
  std::uniform_real_distribution<float> dist(0.0F, 1.0F);

  std::cout << "---------- INGESTING FEATURES ---------\n";
  for (std::size_t i = 0; i < num_nodes; ++i) {
    ggb::Value tensor(feature_dim);
    for (auto& val : tensor) {
      val = dist(engine);
    }

    ggb::Key key{static_cast<std::uint64_t>(i)};
    print_feature_row(key, tensor);
    builder.put_tensor(key, std::move(tensor));
  }
}
}  // namespace

auto main() -> int {
  auto builder = ggb::engine::create_ggb_builder({.db_path = db_path});
  ingest_data(*builder);
  std::cout << "\nFinalizing Store (Building flat file)...\n";
  auto store = builder->build();

  std::cout << "\n------------ GATHER RESULTS -----------\n";
  std::vector<ggb::Key> query_keys = {
      {0}, {1}, {3}, {99}  // 99 tests the 'nullopt' case
  };

  auto results = store->get_multi_tensor(query_keys);
  for (std::size_t i = 0; i < query_keys.size(); ++i) {
    print_feature_row(query_keys[i], results[i]);
  }

  return 0;
}
