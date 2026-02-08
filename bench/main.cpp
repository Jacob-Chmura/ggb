#include <format>
#include <iostream>
#include <string>

#include "config.h"
#include "engines/engines.h"
#include "runner.h"

constexpr std::string dataset_name = "ogbn-arxiv";
constexpr std::string run_id = "run-0001";

auto main() -> int {
  using ggb::bench::RunConfig;
  using ggb::bench::Runner;

  auto cfg = RunConfig::load(dataset_name, run_id);
  if (!cfg) {
    return 1;
  }

  Runner runner(ggb::engine::create_in_memory_builder(), cfg.value());
  auto results = runner.run();
  std::cout << std::format("Number of tensors read: {}",
                           results.num_tensors_read_)
            << std::endl;
  return 0;
}
