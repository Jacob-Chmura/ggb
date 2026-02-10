#include <iostream>
#include <string_view>

#include "config.h"
#include "ggb/core.h"
#include "runner.h"

struct Args {
  std::string_view dataset;
  std::string_view run_id;
  std::string_view engine = "all";
  bool help = false;
};

namespace {

auto print_usage() -> void {
  std::cout << "Usage: bench_main <dataset> <run_id> [options]\n"
            << "Options:\n"
            << "  --engine <mmap|in_memory|all>  (default: all)\n"
            << "  --help                         Show this message\n";
}

auto parse_args(int argc, char** argv) -> std::optional<Args> {
  if (argc < 3) {
    return std::nullopt;
  }

  Args args{.dataset = argv[1], .run_id = argv[2]};

  for (int i = 3; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--engine" && i + 1 < argc) {
      args.engine = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      args.help = true;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return std::nullopt;
    }
  }
  return args;
}

}  // namespace

auto main(int argc, char** argv) -> int {
  const auto args = parse_args(argc, argv);
  if (!args || args->help) {
    print_usage();
    return args->help ? 0 : 1;
  }

  const auto base_cfg =
      ggb::bench::RunConfig::load(args->dataset, args->run_id);
  if (!base_cfg) {
    return 1;
  }

  const auto run_all = (args->engine == "all");
  if (run_all || args->engine == "in_memory") {
    create_runner(ggb::InMemoryConfig{}, *base_cfg).run();
  }
  if (run_all || args->engine == "mmap") {
    create_runner(ggb::FlatMmapConfig{.db_path = "test.ggb"}, *base_cfg).run();
  }

  return 0;
}
