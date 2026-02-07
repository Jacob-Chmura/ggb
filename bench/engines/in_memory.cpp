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

namespace {

class InMemoryFeatureStore final : public FeatureStore {
 public:
  explicit InMemoryFeatureStore(
      std::unordered_map<FeatureStoreKey, FeatureStoreValue,
                         FeatureStoreKeyHash> &&data)
      : data_(std::move(data)) {}

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

  [[nodiscard]] auto get_multi_tensor_async(
      std::span<const FeatureStoreKey> keys) const
      -> std::future<std::vector<std::optional<FeatureStoreValue>>> override {
    std::vector<std::optional<FeatureStoreValue>> results;
    results.reserve(keys.size());

    for (const auto &key : keys) {
      if (auto it = data_.find(key); it != data_.end()) {
        results.emplace_back(it->second);
      } else {
        results.emplace_back(std::nullopt);
      }
    }

    std::promise<std::vector<std::optional<FeatureStoreValue>>> promise;
    promise.set_value(std::move(results));
    return promise.get_future();
  }

 private:
  const std::unordered_map<FeatureStoreKey, FeatureStoreValue,
                           FeatureStoreKeyHash>
      data_;
};

class InMemoryFeatureStoreBuilder final : public FeatureStoreBuilder {
 public:
  auto put_tensor(const FeatureStoreKey &key, const FeatureStoreValue &tensor)
      -> bool override {
    data_[key] = tensor;
    return true;
  }

  auto put_tensor(const FeatureStoreKey &key, FeatureStoreValue &&tensor)
      -> bool override {
    data_[key] = std::move(tensor);
    return true;
  }

  [[nodiscard]] auto build(
      [[maybe_unused]] std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> override {
    return std::make_unique<InMemoryFeatureStore>(std::move(data_));
  }

 private:
  std::unordered_map<FeatureStoreKey, FeatureStoreValue, FeatureStoreKeyHash>
      data_;
};

class InMemoryFeatureEngine : public FeatureEngine {
 public:
  [[nodiscard]] auto create_builder()
      -> std::unique_ptr<FeatureStoreBuilder> override {
    return std::make_unique<InMemoryFeatureStoreBuilder>();
  }

  [[nodiscard]] auto name() const -> std::string_view override { return name_; }

 private:
  static constexpr std::string_view name_ = "InMemoryFeatureEngine";
};

}  // namespace

[[nodiscard]] auto create_in_memory_engine() -> std::unique_ptr<FeatureEngine> {
  return std::make_unique<InMemoryFeatureEngine>();
}

}  // namespace ggb::engine
