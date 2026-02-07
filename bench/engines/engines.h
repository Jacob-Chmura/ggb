#pragma once
#include <memory>

#include "ggb.h"

namespace ggb::engine {

[[nodiscard]] auto create_in_memory_store() -> std::unique_ptr<FeatureStore>;

}  // namespace ggb::engine
