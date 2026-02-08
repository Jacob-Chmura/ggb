#pragma once

#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ggb/core.h"

namespace ggb::engine {

class InMemoryFeatureStore final : public FeatureStore {
 public:
  explicit InMemoryFeatureStore(std::unordered_map<Key, Value, KeyHash> &&data);

  [[nodiscard]] auto name() const -> std::string_view override;
  [[nodiscard]] auto get_num_keys() const -> std::size_t override;
  [[nodiscard]] auto get_tensor_size() const
      -> std::optional<std::size_t> override;
  [[nodiscard]] auto get_multi_tensor_async(std::span<const Key> keys) const
      -> std::future<std::vector<std::optional<Value>>> override;

 private:
  static constexpr std::string_view name_ = "InMemoryFeatureStore";
  const std::unordered_map<Key, Value, KeyHash> data_;
};

class InMemoryFeatureStoreBuilder final : public FeatureStoreBuilder {
 public:
  explicit InMemoryFeatureStoreBuilder(
      [[maybe_unused]] const InMemoryConfig &cfg) {}

  auto put_tensor(const Key &key, const Value &tensor) -> bool override;
  auto put_tensor(const Key &key, Value &&tensor) -> bool override;

  [[nodiscard]] auto build(std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> override;

 private:
  std::unordered_map<Key, Value, KeyHash> data_;
};

}  // namespace ggb::engine
