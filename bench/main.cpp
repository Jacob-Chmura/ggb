#include <iostream>
#include <string>

#include "config.h"
#include "engines/engines.h"
#include "timer.h"

auto main() -> int {
  using ggb::bench::RunConfig;
  using ggb::bench::perf::ScopedTimer;
  {
    const ScopedTimer timer{};
  }
  {
    const ScopedTimer timer{"Foo"};
  }

  const auto store = ggb::engine::create_in_memory_builder()->build();
  std::cout << store->name() << std::endl;

  constexpr std::string dataset_name = "ogbn-arxiv";
  constexpr std::string run_id = "run-0001";

  auto cfg = RunConfig::load_from_run(dataset_name, run_id);
}
