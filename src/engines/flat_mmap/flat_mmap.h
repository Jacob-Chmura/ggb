#pragma once

#include <cstddef>
#include <fstream>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ggb/core.h"

namespace ggb::engine {

class FlatMmapFeatureStore final : public FeatureStore {
 public:
  explicit FlatMmapFeatureStore(
      FlatMmapConfig cfg,
      std::unordered_map<Key, std::size_t, KeyHash>&& key_to_byte,
      std::optional<std::size_t> tensor_size);

  ~FlatMmapFeatureStore() override;

  [[nodiscard]] auto name() const -> std::string_view override;
  [[nodiscard]] auto get_num_keys() const -> std::size_t override;
  [[nodiscard]] auto get_tensor_size() const
      -> std::optional<std::size_t> override;
  [[nodiscard]] auto get_multi_tensor_async(std::span<const Key> keys) const
      -> std::future<std::vector<std::optional<Value>>> override;

 private:
  static constexpr std::string_view name_ = "FlatMmapFeatureStore";
  const FlatMmapConfig cfg_;
  const std::unordered_map<Key, std::size_t, KeyHash> key_to_byte_;
  const std::optional<std::size_t> tensor_size_;

  const float* mapped_data_ = nullptr;
  std::size_t file_size_{0};

  void setup_mmap();
  void cleanup_mmap();
};

class FlatMmapFeatureStoreBuilder final : public FeatureStoreBuilder {
 public:
  explicit FlatMmapFeatureStoreBuilder(const FlatMmapConfig& cfg);

  auto put_tensor(const Key& key, const Value& tensor) -> bool override;
  auto put_tensor(const Key& key, Value&& tensor) -> bool override;

  [[nodiscard]] auto build(std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> override;

 private:
  const FlatMmapConfig cfg_;
  std::ofstream out_file_;
  std::unordered_map<Key, std::size_t, KeyHash> key_to_byte_;
  std::optional<std::size_t> tensor_size_;
  std::size_t write_pos_{0};
};

}  // namespace ggb::engine
