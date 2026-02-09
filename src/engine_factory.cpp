#include <memory>
#include <type_traits>
#include <variant>

#include "common/logging.h"
#include "engines/flat_mmap/flat_mmap.h"
#include "engines/in_memory/in_memory.h"
#include "ggb/core.h"

namespace ggb {

auto create_builder(const EngineConfig& cfg)
    -> std::unique_ptr<FeatureStoreBuilder> {
  return std::visit(
      [](auto&& arg) -> std::unique_ptr<FeatureStoreBuilder> {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, FlatMmapConfig>) {
          GGB_LOG_DEBUG("Creating FlatMmap builder");
          return std::make_unique<engine::FlatMmapFeatureStoreBuilder>(arg);
        } else if constexpr (std::is_same_v<T, InMemoryConfig>) {
          GGB_LOG_DEBUG("Creating InMemory builder");
          return std::make_unique<engine::InMemoryFeatureStoreBuilder>(arg);
        }
      },
      cfg);
}
}  // namespace ggb
