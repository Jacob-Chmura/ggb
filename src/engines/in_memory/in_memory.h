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
  explicit InMemoryFeatureStore(
      std::vector<float> &&blob,
      std::unordered_map<Key, std::size_t, KeyHash> &&offsets,
      std::optional<std::size_t> tensor_size);

  [[nodiscard]] auto name() const -> std::string_view override;
  [[nodiscard]] auto get_num_keys() const -> std::size_t override;
  [[nodiscard]] auto get_tensor_size() const
      -> std::optional<std::size_t> override;
  [[nodiscard]] auto get_multi_tensor_async(std::span<const Key> keys) const
      -> std::future<std::vector<std::optional<Value>>> override;

 private:
  static constexpr std::string_view name_ = "InMemoryFeatureStore";
  const std::vector<float> blob_;
  const std::unordered_map<Key, std::size_t, KeyHash> offsets_;
  const std::optional<std::size_t> tensor_size_;
};

class InMemoryFeatureStoreBuilder final : public FeatureStoreBuilder {
 public:
  explicit InMemoryFeatureStoreBuilder(
      [[maybe_unused]] const InMemoryConfig &cfg) {}

  auto put_tensor_impl(const Key &key, const Value &tensor) -> bool override;
  auto put_tensor_impl(const Key &key, Value &&tensor) -> bool override;

  [[nodiscard]] auto build_impl(
      std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> override;

 private:
  std::vector<float> blob_;
  std::unordered_map<Key, std::size_t, KeyHash> offsets_;
  std::optional<std::size_t> tensor_size_;
};

}  // namespace ggb::engine
