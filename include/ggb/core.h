#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ggb {

struct FlatMmapConfig {
  std::string db_path;
};

struct InMemoryConfig {};

using EngineConfig = std::variant<FlatMmapConfig, InMemoryConfig>;

using NodeID = std::uint64_t;
using Value = std::vector<float>;

struct Key {
  // TODO(kuba) For now, we only support homogenous node keys
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

// TODO(kuba): for now, we just have edge list
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

  auto put_tensor(const Key &key, const Value &tensor) -> bool {
    check_not_built();
    return put_tensor_impl(key, tensor);
  }

  auto put_tensor(const Key &key, Value &&tensor) -> bool {
    check_not_built();
    return put_tensor_impl(key, std::move(tensor));
  }

  auto build(std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> {
    check_not_built();
    is_built_ = true;
    return build_impl(graph);
  }

 protected:
  virtual auto put_tensor_impl(const Key &key, const Value &tensor) -> bool = 0;
  virtual auto put_tensor_impl(const Key &key, Value &&tensor) -> bool = 0;
  virtual auto build_impl(std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> = 0;

 private:
  auto check_not_built() const -> void {
    if (is_built_) {
      throw std::runtime_error(
          "GGB Error: FeatureStoreBuilder is defunct. After calling `build`, "
          "futher calls to `put_tensor` or `build` are prohibited.");
    }
  }

  bool is_built_{false};
};

auto create_builder(const EngineConfig &cfg)
    -> std::unique_ptr<FeatureStoreBuilder>;

}  // namespace ggb
