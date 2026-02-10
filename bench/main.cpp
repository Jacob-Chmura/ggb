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

  auto base_cfg = RunConfig::load(dataset_name, run_id);
  if (!base_cfg) {
    return 1;
  }

  create_runner(ggb::InMemoryConfig{}, *base_cfg).run();
  create_runner(ggb::FlatMmapConfig{.db_path = "test.ggb"}, *base_cfg).run();

  return 0;
}
