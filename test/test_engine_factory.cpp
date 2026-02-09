#include "engines/flat_mmap/flat_mmap.h"
#include "engines/in_memory/in_memory.h"
#include "ggb/core.h"

// Third-party
#include <gtest/gtest.h>

using namespace ggb;

TEST(EngineFactory, CreateInMemoryBuilder) {
  const EngineConfig cfg = InMemoryConfig{};
  auto builder = create_builder(cfg);

  auto* ptr = dynamic_cast<engine::InMemoryFeatureStoreBuilder*>(builder.get());
  EXPECT_NE(ptr, nullptr)
      << "Factory failed to return InMemoryFeatureStoreBuilder for "
         "InMemoryConfig";

  EXPECT_EQ(1, 1);
}

TEST(EngineFactory, CreateFlatMmapBuilder) {
  const EngineConfig cfg = FlatMmapConfig{.db_path = {"/tmp/foo.ggb"}};
  auto builder = create_builder(cfg);

  auto* ptr = dynamic_cast<engine::FlatMmapFeatureStoreBuilder*>(builder.get());
  EXPECT_NE(ptr, nullptr) << "Factory failed to return "
                             "FlatMmapFeatureStoreBuilder for FlatMmapConfig";
}
