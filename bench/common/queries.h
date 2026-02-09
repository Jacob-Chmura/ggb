#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/logging.h"
#include "ggb/core.h"

namespace ggb::bench {

class QueryLoader {
  using Query = std::vector<ggb::Key>;

 public:
  static auto from_csv(const std::string& path) -> std::vector<Query> {
    std::vector<Query> queries;
    std::ifstream file(path);
    if (!file.is_open()) {
      GGB_LOG_ERROR("QueryLoader: Could not open {}", path);
      throw std::runtime_error("Failed to open query file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string part;
      Query query{};

      while (std::getline(ss, part, ',')) {
        try {
          query.push_back(ggb::Key{.NodeID = std::stoull(part)});
        } catch (...) {
          GGB_LOG_WARN("QueryLoader: Skipping invalid NodeID '{}' in {}", part,
                       path);
        }
      }

      queries.push_back(query);
    }
    GGB_LOG_INFO("Loaded {} queries from {}", queries.size(), path);
    return queries;
  }
};
}  // namespace ggb::bench
