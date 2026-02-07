#pragma once
#include <memory>

#include "ggb.h"

namespace ggb::engine {

[[nodiscard]] auto create_in_memory_builder()
    -> std::unique_ptr<FeatureStoreBuilder>;

}  // namespace ggb::engine
