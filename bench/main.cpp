#include <string>

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

  {
    Runner runner(ggb::create_builder(ggb::InMemoryConfig{}), cfg.value());
    auto results = runner.run();
    results.print();
  }

  {
    Runner runner(
        ggb::create_builder(ggb::FlatMmapConfig{.db_path = "test.ggb"}),
        cfg.value());
    auto results = runner.run();
    results.print();
  }

  return 0;
}
