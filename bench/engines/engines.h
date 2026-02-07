#pragma once
#include <memory>

#include "ggb.h"

namespace ggb::engine {

[[nodiscard]] auto create_in_memory_engine() -> std::unique_ptr<FeatureEngine>;

}  // namespace ggb::engine
