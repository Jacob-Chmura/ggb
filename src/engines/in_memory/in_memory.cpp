#include "in_memory.h"

#include <cstddef>
#include <future>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/logging.h"

namespace ggb::engine {

InMemoryFeatureStore::InMemoryFeatureStore(
    std::vector<float> &&blob,
    std::unordered_map<Key, std::size_t, KeyHash> &&offsets,
    std::optional<std::size_t> tensor_size)
    : blob_(std::move(blob)),
      offsets_(std::move(offsets)),
      tensor_size_(tensor_size) {}

[[nodiscard]] auto InMemoryFeatureStore::name() const -> std::string_view {
  return name_;
}

[[nodiscard]] auto InMemoryFeatureStore::get_num_keys() const -> std::size_t {
  return offsets_.size();
}

[[nodiscard]] auto InMemoryFeatureStore::get_tensor_size() const
    -> std::optional<std::size_t> {
  return tensor_size_;
}

[[nodiscard]] auto InMemoryFeatureStore::get_multi_tensor_async(
    std::span<const Key> keys) const
    -> std::future<std::vector<std::optional<Value>>> {
  std::vector<std::optional<Value>> results;
  if (!tensor_size_.has_value()) {
    GGB_LOG_WARN("Empty tensor dimension found");
    results.assign(keys.size(), std::nullopt);
  } else {
    results.reserve(keys.size());

    for (const auto &key : keys) {
      if (auto it = offsets_.find(key); it != offsets_.end()) {
        const float *start = blob_.data() + it->second;
        results.emplace_back(Value(start, start + tensor_size_.value()));
      } else {
        results.emplace_back(std::nullopt);
      }
    }
  }

  std::promise<std::vector<std::optional<Value>>> promise;
  promise.set_value(std::move(results));
  return promise.get_future();
}

auto InMemoryFeatureStoreBuilder::put_tensor_impl(const Key &key,
                                                  const Value &tensor) -> bool {
  if (!tensor_size_.has_value()) {
    tensor_size_ = tensor.size();
    constexpr auto num_nodes_to_reserve = 10'000;
    blob_.reserve(tensor_size_.value() * num_nodes_to_reserve);
  }

  if (tensor.size() != tensor_size_.value()) {
    GGB_LOG_ERROR("Mismatched tensor size: got {}, expected {}", tensor.size(),
                  tensor_size_.value());
    return false;
  }

  offsets_[key] = blob_.size();
  blob_.insert(blob_.end(), tensor.begin(), tensor.end());
  return true;
}

auto InMemoryFeatureStoreBuilder::put_tensor_impl(const Key &key,
                                                  Value &&tensor) -> bool {
  return put_tensor_impl(key, static_cast<const Value &>(tensor));
}

[[nodiscard]] auto InMemoryFeatureStoreBuilder::build_impl(
    [[maybe_unused]] std::optional<GraphTopology> graph)
    -> std::unique_ptr<FeatureStore> {
  GGB_LOG_INFO(
      "Building InMemoryStore\n\tTotal Keys: {}\n\tEst. Memory: {:.3f} GB",
      offsets_.size(),
      static_cast<double>(blob_.size() * sizeof(float)) / (1024 * 1024 * 1024));
  return std::make_unique<InMemoryFeatureStore>(
      std::move(blob_), std::move(offsets_), tensor_size_);
}
}  // namespace ggb::engine
