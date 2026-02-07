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

class InMemoryFeatureStore;

class InMemoryFeatureStoreBuilder final : public FeatureStoreBuilder {
 public:
  explicit InMemoryFeatureStoreBuilder(InMemoryFeatureStore &parent)
      : parent_(parent) {}

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

  auto commit([[maybe_unused]] std::optional<GraphTopology> graph =
                  std::nullopt) -> void override;

 private:
  InMemoryFeatureStore &parent_;
  std::unordered_map<FeatureStoreKey, FeatureStoreValue, FeatureStoreKeyHash>
      data_;
};

class InMemoryFeatureStore final : public FeatureStore {
 public:
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

  [[nodiscard]] auto create_builder()
      -> std::unique_ptr<FeatureStoreBuilder> override {
    return std::make_unique<InMemoryFeatureStoreBuilder>(*this);
  }

 private:
  static constexpr std::string_view name_ = "InMemoryFeatureStore";
  std::unordered_map<FeatureStoreKey, FeatureStoreValue, FeatureStoreKeyHash>
      data_;

  friend class InMemoryFeatureStoreBuilder;
};

inline auto InMemoryFeatureStoreBuilder::commit(
    [[maybe_unused]] std::optional<GraphTopology> graph) -> void {
  // Merge the buffer into the parent's data
  for (auto &[key, value] : data_) {
    parent_.data_[key] = std::move(value);
  }
  data_.clear();
}

}  // namespace

[[nodiscard]] auto create_in_memory_store() -> std::unique_ptr<FeatureStore> {
  return std::make_unique<InMemoryFeatureStore>();
}

}  // namespace ggb::engine
