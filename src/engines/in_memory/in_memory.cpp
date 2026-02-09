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
    std::unordered_map<Key, Value, KeyHash> &&data)
    : data_(std::move(data)) {}

[[nodiscard]] auto InMemoryFeatureStore::name() const -> std::string_view {
  return name_;
}

[[nodiscard]] auto InMemoryFeatureStore::get_num_keys() const -> std::size_t {
  return data_.size();
}

[[nodiscard]] auto InMemoryFeatureStore::get_tensor_size() const
    -> std::optional<std::size_t> {
  if (data_.empty()) {
    return std::nullopt;
  }
  return data_.begin()->second.size();
}

[[nodiscard]] auto InMemoryFeatureStore::get_multi_tensor_async(
    std::span<const Key> keys) const
    -> std::future<std::vector<std::optional<Value>>> {
  std::vector<std::optional<Value>> results;
  results.reserve(keys.size());

  for (const auto &key : keys) {
    if (auto it = data_.find(key); it != data_.end()) {
      results.emplace_back(it->second);
    } else {
      results.emplace_back(std::nullopt);
    }
  }

  std::promise<std::vector<std::optional<Value>>> promise;
  promise.set_value(std::move(results));
  return promise.get_future();
}

auto InMemoryFeatureStoreBuilder::put_tensor(const Key &key,
                                             const Value &tensor) -> bool {
  data_[key] = tensor;
  return true;
}

auto InMemoryFeatureStoreBuilder::put_tensor(const Key &key, Value &&tensor)
    -> bool {
  data_[key] = std::move(tensor);
  return true;
}

[[nodiscard]] auto InMemoryFeatureStoreBuilder::build(
    [[maybe_unused]] std::optional<GraphTopology> graph)
    -> std::unique_ptr<FeatureStore> {
  auto estimated_bytes = data_.size() * (sizeof(Key) + sizeof(Value));
  GGB_LOG_INFO(
      "Building InMemoryStore\n\tTotal Keys: {}\n\tEst. Memory: {:.3f} GB",
      data_.size(),
      static_cast<double>(estimated_bytes) / (1024 * 1024 * 1024));
  return std::make_unique<InMemoryFeatureStore>(std::move(data_));
}

}  // namespace ggb::engine
