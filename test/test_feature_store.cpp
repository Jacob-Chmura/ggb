
// #include <cstddef>
// #include <cstdint>
// #include <iostream>
// #include <optional>
// #include <random>
// #include <sstream>
// #include <stdexcept>
// #include <string>
// #include <utility>
// #include <vector>
//
// #include "common/logging.h"
// #include "ggb/core.h"
//
//// Third-party
// #include <gtest/gtest.h>
//
// constexpr std::string db_path = "test.ggb";
// const std::size_t num_nodes = 5;
// const std::size_t feature_dim = 3;
//
// namespace {
// void print_feature_row(ggb::Key key, const std::optional<ggb::Value>& feat) {
//   if (!feat.has_value()) {
//     GGB_LOG_WARN("Node: {} | Feat: [Not Found]", key.NodeID);
//     return;
//   }
//
//   std::ostringstream oss;
//   oss << "[";
//   for (std::size_t i = 0; i < feat->size(); ++i) {
//     oss << (*feat)[i] << (i == feat->size() - 1 ? "" : ", ");
//   }
//   oss << "]";
//
//   GGB_LOG_INFO("Node: {} | Feat: {}", key.NodeID, oss.str());
// }
//
// void ingest_data(ggb::FeatureStoreBuilder& builder) {
//   std::random_device rng;
//   std::mt19937 engine(rng());
//   std::uniform_real_distribution<float> dist(0.0F, 1.0F);
//
//   GGB_LOG_INFO("---------- INGESTING FEATURES ---------");
//   for (std::size_t i = 0; i < num_nodes; ++i) {
//     ggb::Value tensor(feature_dim);
//     for (auto& val : tensor) {
//       val = dist(engine);
//     }
//
//     const ggb::Key key{static_cast<std::uint64_t>(i)};
//     print_feature_row(key, tensor);
//     builder.put_tensor(key, std::move(tensor));
//   }
// }
//
// auto test_with_builder(std::unique_ptr<ggb::FeatureStoreBuilder> builder)
//     -> void {
//   ingest_data(*builder);
//   auto store = builder->build();
//
//   std::vector<ggb::Key> query_keys = {
//       {0}, {1}, {3}, {99}  // 99 tests the 'nullopt' case
//   };
//   auto results = store->get_multi_tensor(query_keys);
//
//   GGB_LOG_INFO("---------- GATHER RESULTS ---------");
//   for (std::size_t i = 0; i < query_keys.size(); ++i) {
//     print_feature_row(query_keys[i], results[i]);
//   }
// }
// }  // namespace
//
// auto main() -> int {
//   test_with_builder(
//       ggb::create_builder(ggb::FlatMmapConfig{.db_path = db_path}));
//   test_with_builder(ggb::create_builder(ggb::InMemoryConfig{}));
//
//   // Attempt `put` after `build` should fail
//   {
//     auto builder = ggb::create_builder(ggb::InMemoryConfig{});
//     builder->build();
//     try {
//       builder->put_tensor(ggb::Key{.NodeID = 0}, ggb::Value{});
//     } catch (const std::runtime_error& e) {
//       GGB_LOG_WARN("Caught expected error: {}", e.what());
//     }
//   }
//
//   // Attempt `build` after `build` should also fail
//   {
//     auto builder = ggb::create_builder(ggb::InMemoryConfig{});
//     builder->build();
//     try {
//       builder->build();
//     } catch (const std::runtime_error& e) {
//       GGB_LOG_WARN("Caught expected error: {}", e.what());
//     }
//   }
//   return 0;
// }
