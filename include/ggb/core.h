#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ggb {

struct GGBConfig {
  std::string db_path;
};

struct InMemoryConfig {};

using EngineConfig = std::variant<GGBConfig, InMemoryConfig>;

using NodeID = std::uint64_t;
using Value = std::vector<float>;

struct Key {
  // For now, we only support homogenous node keys
  std::uint64_t NodeID;

  auto operator<=>(const Key &) const = default;

  friend auto operator<<(std::ostream &os, const Key &key) -> std::ostream & {
    return os << "NodeID(" << key.NodeID << ")";
  }
};

struct KeyHash {
  auto operator()(const Key &key) const noexcept -> std::size_t {
    return key.NodeID;
  }
};

struct GraphTopology {
  std::span<const std::pair<NodeID, NodeID>> edges;
};

class FeatureStore {
 public:
  virtual ~FeatureStore() = default;

  [[nodiscard]] virtual auto name() const -> std::string_view = 0;
  [[nodiscard]] virtual auto get_num_keys() const -> std::size_t = 0;
  [[nodiscard]] virtual auto get_tensor_size() const
      -> std::optional<std::size_t> = 0;

  [[nodiscard]] virtual auto get_multi_tensor_async(std::span<const Key> keys)
      const -> std::future<std::vector<std::optional<Value>>> = 0;

  [[nodiscard]] auto get_multi_tensor(std::span<const Key> keys) const
      -> std::vector<std::optional<Value>> {
    return get_multi_tensor_async(keys).get();
  }
};

class FeatureStoreBuilder {
 public:
  virtual ~FeatureStoreBuilder() = default;

  virtual auto put_tensor(const Key &key, const Value &tensor) -> bool = 0;
  virtual auto put_tensor(const Key &key, Value &&tensor) -> bool = 0;
  virtual auto build(std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> = 0;
};

auto create_builder(const EngineConfig &cfg)
    -> std::unique_ptr<FeatureStoreBuilder>;

}  // namespace ggb
