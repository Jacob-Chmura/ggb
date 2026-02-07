#pragma once

#include <string>
#include <vector>

#include "ggb.h"

namespace ggb::bench {

struct Query {
  std::vector<ggb::Key> keys;
};

class QueryLoader {
 public:
  static auto from_csv(const std::string& path) -> std::vector<Query> {
    std::vector<Query> queries;
    // std::ifstream file(path);
    return queries;
  }
};

}  // namespace ggb::bench
