#include <iostream>

#include "engines/engines.h"
#include "timer.h"

auto main() -> int {
  const Timer timer{};
  const auto engine = ggb::engine::create_in_memory_store();
  std::cout << engine->name() << std::endl;
}
