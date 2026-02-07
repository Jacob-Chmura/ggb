#include <iostream>

#include "engines/engines.h"
#include "timer.h"

auto main() -> int {
  {
    const ggb::perf::ScopedTimer timer{};
  }
  {
    const ggb::perf::ScopedTimer timer{"Foo"};
  }

  const auto store = ggb::engine::create_in_memory_builder()->build();
  std::cout << store->name() << std::endl;
}
