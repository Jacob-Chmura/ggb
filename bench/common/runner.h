#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "config.h"
#include "ggb/core.h"
#include "queries.h"
#include "result.h"
#include "timer.h"

namespace ggb::bench {
class Runner {
 public:
  explicit Runner(std::unique_ptr<FeatureStoreBuilder> builder, RunConfig cfg)
      : builder_(std::move(builder)), cfg_(std::move(cfg)) {}

  auto run() -> perf::BenchResult {
    perf::BenchResult result{.cfg = cfg_, .latencies_us_{}};

    {
      const perf::ScopedTimer timer("Ingestion");
      ingest_features();
      ingest_graph();
    }

    {
      const perf::ScopedTimer timer("Building");
      store_ = builder_->build(graph_);
    }

    // Clear edge buffer to free some RAM
    edge_buffer_.clear();
    edge_buffer_.shrink_to_fit();

    {
      run_queries(result);
    }

    return result;
  }

 private:
  std::unique_ptr<FeatureStoreBuilder> builder_;
  std::unique_ptr<FeatureStore> store_;
  RunConfig cfg_{};
  std::optional<ggb::GraphTopology> graph_;
  std::vector<std::pair<ggb::NodeID, ggb::NodeID>> edge_buffer_;

  auto ingest_graph() -> void {
    std::ifstream file(cfg_.edge_list_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open edge list: " +
                               cfg_.edge_list_path.string());
    }

    std::string line;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string part;
      std::vector<ggb::NodeID> nodes;

      while (std::getline(ss, part, ',')) {
        nodes.push_back(std::stoull(part));
      }

      if (nodes.size() != 2) {
        throw std::runtime_error(
            "Malformed edge list: expected 2 nodes per line, got " +
            std::to_string(nodes.size()));
      }
      edge_buffer_.emplace_back(nodes[0], nodes[1]);
    }
    graph_ = ggb::GraphTopology{.edges = std::span(edge_buffer_)};
  }

  auto ingest_features() -> void {
    auto fd = open(cfg_.node_feat_path.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Failed to open feature file");
    }

    struct stat st;
    fstat(fd, &st);
    const auto file_size = st.st_size;

    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
      throw std::runtime_error("mmap failed");
    }

    // Hint to the kernel that we will read this start-to-finish
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    const char* ptr = static_cast<const char*>(mapped);
    const char* end = ptr + file_size;
    std::uint64_t node_id{0};

    ggb::Value tensor;
    tensor.reserve(128);  // Pre-reserve typical GNN feature dim

    while (ptr < end) {
      tensor.clear();
      char* next_ptr = nullptr;

      // Inner loop: parse floats until we hit a newline
      while (ptr < end && *ptr != '\n' && *ptr != '\r') {
        const float val = std::strtof(ptr, &next_ptr);
        if (ptr == next_ptr) {
          break;  // Could not parse a number
        }

        tensor.push_back(val);
        ptr = next_ptr;

        // Skip the comma if present
        if (ptr < end && *ptr == ',') {
          ptr++;
        }
      }

      if (!tensor.empty()) {
        builder_->put_tensor({node_id++}, tensor);
      }

      // Advance to the start of the next line
      while (ptr < end && (*ptr == '\n' || *ptr == '\r')) {
        ptr++;
      }
    }

    munmap(mapped, file_size);
  }

  auto run_queries(perf::BenchResult& result) -> void {
    for (const auto& query_csv : cfg_.query_csvs) {
      const auto queries = QueryLoader::from_csv(query_csv.string());

      for (const auto& query : queries) {
        {
          const perf::ScopedTimer timer(
              [&](std::uint64_t us) { result.latencies_us_.push_back(us); });
          auto feats = store_->get_multi_tensor(std::span(query));
        }

        result.num_tensors_read_ += query.size();
      }

      // TODO(kuba): more metrics and properties across seeds
      break;
    }
  }
};
}  // namespace ggb::bench
