#include <string>

#include "config.h"
#include "ggb/core.h"
#include "runner.h"

// TODO(kuba): Pass args via cli
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
    auto engine_cfg = ggb::InMemoryConfig{};
    cfg->engine = engine_cfg;
    Runner runner(ggb::create_builder(engine_cfg), cfg.value());
    runner.run();
  }

  {
    auto engine_cfg = ggb::FlatMmapConfig{.db_path = "test.ggb"};
    cfg->engine = engine_cfg;
    Runner runner(ggb::create_builder(engine_cfg), cfg.value());
    runner.run();
  }

  return 0;
}
