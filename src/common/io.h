#pragma once

#include <sys/mman.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "common/logging.h"
#include "common/mmap_region.h"
#include "ggb/core.h"

namespace ggb::io {
inline auto ingest_features_from_csv(const std::string& path,
                                     FeatureStoreBuilder& builder) -> void {
  const detail::MmapRegion mmap(path);

  // Hint to the kernel that we will read this start-to-finish
  mmap.advise(MADV_SEQUENTIAL);

  const char* ptr = static_cast<const char*>(mmap.data());
  const char* const end = ptr + mmap.size();

  std::uint64_t node_id{0};
  ggb::Value tensor;
  tensor.reserve(128);  // Pre-reserve typical GNN feature dim

  while (ptr < end) {
    tensor.clear();
    char* next_ptr = nullptr;

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
      builder.put_tensor({node_id++}, tensor);
    }

    while (ptr < end && (*ptr == '\n' || *ptr == '\r')) {
      ptr++;
    }
  }

  GGB_LOG_INFO("Ingested {} node features from {}", node_id, path);
}

inline void ingest_edgelist_from_csv(
    const std::string& path,
    std::vector<std::pair<ggb::NodeID, ggb::NodeID>>& out_buffer) {
  const detail::MmapRegion mmap(path);
  mmap.advise(MADV_SEQUENTIAL);

  const char* ptr = static_cast<const char*>(mmap.data());
  const char* const end = ptr + mmap.size();

  while (ptr < end) {
    char* next_ptr = nullptr;

    // Parse Source Node
    const auto src = std::strtoull(ptr, &next_ptr, 10);
    if (ptr == next_ptr) {
      break;
    }
    ptr = next_ptr;

    // Skip comma
    if (ptr < end && *ptr == ',') {
      ptr++;
    }

    // Parse Destination Node
    const auto dst = std::strtoull(ptr, &next_ptr, 10);
    ptr = next_ptr;

    out_buffer.emplace_back(src, dst);

    while (ptr < end && (*ptr == '\n' || *ptr == '\r')) {
      ptr++;
    }
  }

  GGB_LOG_INFO("Ingested {} edges from {}", out_buffer.size(), path);
}
}  // namespace ggb::io
