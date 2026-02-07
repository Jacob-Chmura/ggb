#include <cstddef>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ggb.h"

namespace ggb::engine {

namespace in_memory {

class InMemoryFeatureStore final : public FeatureStore {
 public:
  explicit InMemoryFeatureStore(std::unordered_map<Key, Value, KeyHash> &&data)
      : data_(std::move(data)) {}

  [[nodiscard]] auto name() const -> std::string_view override { return name_; }

  [[nodiscard]] auto get_num_keys() const -> std::size_t override {
    return data_.size();
  }
  [[nodiscard]] auto get_tensor_size() const
      -> std::optional<std::size_t> override {
    if (data_.empty()) {
      return std::nullopt;
    }
    return data_.begin()->second.size();
  }

  [[nodiscard]] auto get_multi_tensor_async(std::span<const Key> keys) const
      -> std::future<std::vector<std::optional<Value>>> override {
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

 private:
  static constexpr std::string_view name_ = "InMemoryFeatureStore";
  const std::unordered_map<Key, Value, KeyHash> data_;
};

class InMemoryFeatureStoreBuilder final : public FeatureStoreBuilder {
 public:
  auto put_tensor(const Key &key, const Value &tensor) -> bool override {
    data_[key] = tensor;
    return true;
  }

  auto put_tensor(const Key &key, Value &&tensor) -> bool override {
    data_[key] = std::move(tensor);
    return true;
  }

  [[nodiscard]] auto build(
      [[maybe_unused]] std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> override {
    return std::make_unique<InMemoryFeatureStore>(std::move(data_));
  }

 private:
  std::unordered_map<Key, Value, KeyHash> data_;
};

}  // namespace in_memory

[[nodiscard]] auto create_in_memory_builder()
    -> std::unique_ptr<FeatureStoreBuilder> {
  return std::make_unique<in_memory::InMemoryFeatureStoreBuilder>();
}

}  // namespace ggb::engine
