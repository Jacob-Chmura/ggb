#include <string>
#include <utility>

#include "config.h"
#include "ggb/core.h"
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

  auto builder = ggb::create_builder(ggb::InMemoryConfig{});

  Runner runner(std::move(builder), cfg.value());
  auto results = runner.run();
  results.print();
  return 0;
}
