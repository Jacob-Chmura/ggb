#include <filesystem>
#include <vector>

#include "engines/flat_mmap/flat_mmap.h"
#include "engines/in_memory/in_memory.h"
#include "ggb/core.h"

// Third-party
#include <gtest/gtest.h>

namespace {

template <typename TBuilder, typename TConfig>
auto test_builder(const TConfig& cfg) -> void {
  TBuilder builder(cfg);

  EXPECT_TRUE(builder.put_tensor({0}, {1.0, 2.0}));

  // Test: Mismatched dimensions
  EXPECT_FALSE(builder.put_tensor({1}, {3.0, 4.0, 5.0}));

  const auto store = builder.build();
  ASSERT_NE(store, nullptr);

  // Attempt `put_tensor` after `build` should throw
  EXPECT_THROW({ builder.put_tensor({2}, {5.0, 6.0}); }, std::runtime_error);

  // Attempt `build` after `build` should also throw
  EXPECT_THROW({ builder.build(); }, std::runtime_error);
}

template <typename TBuilder, typename TConfig>
void test_store(const TConfig& cfg) {
  TBuilder builder(cfg);
  const std::vector<float> feat0 = {1.0, 2.0};
  const std::vector<float> feat1 = {3.0, 4.0};

  builder.put_tensor({0}, feat0);
  builder.put_tensor({1}, feat1);
  const auto store = builder.build();

  // Metadata Verification
  EXPECT_FALSE(store->name().empty());
  EXPECT_EQ(store->get_num_keys(), 2);
  ASSERT_TRUE(store->get_tensor_size().has_value());
  EXPECT_EQ(store->get_tensor_size().value(), 2);

  // Data Retrieval (Async)
  const std::vector<ggb::Key> keys = {{0}, {1}, {2}};
  auto future = store->get_multi_tensor_async(keys);
  const auto results = future.get();

  ASSERT_EQ(results.size(), 3);

  // Key 0: Exists
  ASSERT_TRUE(results[0].has_value());
  EXPECT_FLOAT_EQ(results[0].value()[0], 1.0);

  // Key 1: Exists
  ASSERT_TRUE(results[1].has_value());
  EXPECT_FLOAT_EQ(results[1].value()[1], 4.0);

  // Key 2: Missing
  EXPECT_FALSE(results[2].has_value());

  // Data Retrieval (Sync)
  const std::vector<ggb::Key> sync_keys = {{0}};
  const auto sync_results = store->get_multi_tensor(keys);
  ASSERT_EQ(sync_results, results);
}

}  // namespace

// --- In-Memory Tests ---

TEST(InMemoryFeatureStore, BuilderTest) {
  const ggb::InMemoryConfig cfg{};
  test_builder<ggb::engine::InMemoryFeatureStoreBuilder>(cfg);
}

TEST(InMemoryFeatureStore, RetrievalTest) {
  const ggb::InMemoryConfig cfg{};
  test_store<ggb::engine::InMemoryFeatureStoreBuilder>(cfg);
}

// --- FlatMmap Tests ---

TEST(FlatMmapFeatureStore, BuilderTest) {
  const ggb::FlatMmapConfig cfg{.db_path = "test.ggb"};
  test_builder<ggb::engine::FlatMmapFeatureStoreBuilder>(cfg);
  std::filesystem::remove(cfg.db_path);
}

TEST(FlatMmapFeatureStore, RetrievalTest) {
  const ggb::FlatMmapConfig cfg{.db_path = "test.ggb"};
  test_store<ggb::engine::FlatMmapFeatureStoreBuilder>(cfg);
  std::filesystem::remove(cfg.db_path);
}
