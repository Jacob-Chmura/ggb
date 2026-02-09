#include "engines/flat_mmap/flat_mmap.h"
#include "engines/in_memory/in_memory.h"
#include "ggb/core.h"

// Third-party
#include <gtest/gtest.h>

TEST(EngineFactory, CreateInMemoryBuilder) {
  const auto cfg = ggb::InMemoryConfig{};
  auto builder = create_builder(cfg);
  auto* ptr =
      dynamic_cast<ggb::engine::InMemoryFeatureStoreBuilder*>(builder.get());
  EXPECT_NE(ptr, nullptr)
      << "Factory failed to return InMemoryFeatureStoreBuilder for "
         "InMemoryConfig";
}

TEST(EngineFactory, CreateFlatMmapBuilder) {
  const auto cfg = ggb::FlatMmapConfig{.db_path = {"/tmp/foo.ggb"}};
  auto builder = create_builder(cfg);
  auto* ptr =
      dynamic_cast<ggb::engine::FlatMmapFeatureStoreBuilder*>(builder.get());
  EXPECT_NE(ptr, nullptr) << "Factory failed to return "
                             "FlatMmapFeatureStoreBuilder for FlatMmapConfig";
}
