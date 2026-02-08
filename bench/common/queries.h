#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "ggb.h"

namespace ggb::bench {

class QueryLoader {
  using Query = std::vector<ggb::Key>;

 public:
  static auto from_csv(const std::string& path) -> std::vector<Query> {
    std::vector<Query> queries;
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open query file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string part;
      Query query{};

      while (std::getline(ss, part, ',')) {
        query.push_back(ggb::Key{.NodeID = std::stoull(part)});
      }

      queries.push_back(query);
    }
    return queries;
  }
};
}  // namespace ggb::bench
