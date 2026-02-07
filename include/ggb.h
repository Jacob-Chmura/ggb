#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ggb {

using NodeID = std::uint64_t;
using FeatureStoreValue = std::vector<float>;

struct FeatureStoreKey {
  // For now, we only support homogenous node keys
  std::uint64_t NodeID;

  auto operator<=>(const FeatureStoreKey &) const = default;
};

struct FeatureStoreKeyHash {
  auto operator()(const FeatureStoreKey &key) const noexcept -> std::size_t {
    return key.NodeID;
  }
};

struct GraphTopology {
  std::span<const std::pair<NodeID, NodeID>> edges;
};

class FeatureStore {
 public:
  virtual ~FeatureStore() = default;

  [[nodiscard]] virtual auto get_num_keys() const -> std::size_t = 0;
  [[nodiscard]] virtual auto get_tensor_size() const
      -> std::optional<std::size_t> = 0;

  [[nodiscard]] virtual auto get_multi_tensor_async(
      std::span<const FeatureStoreKey> keys) const
      -> std::future<std::vector<std::optional<FeatureStoreValue>>> = 0;

  [[nodiscard]] auto get_multi_tensor(std::span<const FeatureStoreKey> keys)
      const -> std::vector<std::optional<FeatureStoreValue>> {
    return get_multi_tensor_async(keys).get();
  }
};

class FeatureStoreBuilder {
 public:
  virtual ~FeatureStoreBuilder() = default;

  virtual auto put_tensor(const FeatureStoreKey &key,
                          const FeatureStoreValue &tensor) -> bool = 0;
  virtual auto put_tensor(const FeatureStoreKey &key,
                          FeatureStoreValue &&tensor) -> bool = 0;

  [[nodiscard]] virtual auto build(
      std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> = 0;
};

class FeatureEngine {
 public:
  virtual ~FeatureEngine() = default;
  [[nodiscard]] virtual auto create_builder()
      -> std::unique_ptr<FeatureStoreBuilder> = 0;

  [[nodiscard]] virtual auto name() const -> std::string_view = 0;
};

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

struct Config {
  std::string db_path;
};

struct Context {
  Config config;
};

class OldFeatureStore {
 public:
  virtual ~OldFeatureStore() = default;

  [[nodiscard]] virtual auto num_nodes() const -> std::size_t = 0;
  [[nodiscard]] virtual auto feature_dim() const -> std::size_t = 0;
  [[nodiscard]] virtual auto get_feature(NodeID node) const
      -> std::span<const float> = 0;
};

class InMemoryOldFeatureStore final : public OldFeatureStore {
 public:
  explicit InMemoryOldFeatureStore(std::vector<std::vector<float>> data)
      : data_(std::move(data)),
        num_nodes_(data_.size()),
        feature_dim_(data_.empty() ? 0 : data_[0].size()) {}

  [[nodiscard]] auto num_nodes() const -> std::size_t override {
    return num_nodes_;
  }
  [[nodiscard]] auto feature_dim() const -> std::size_t override {
    return feature_dim_;
  }

  [[nodiscard]] auto get_feature(NodeID node) const
      -> std::span<const float> override {
    return {data_[node]};
  }

 private:
  std::vector<std::vector<float>> data_;
  std::size_t num_nodes_ = 0;
  std::size_t feature_dim_ = 0;
};

namespace detail {
inline auto write_file(const std::string &db_path,
                       const OldFeatureStore &features) -> void {
  std::cout << "Building graph at db_path: " << db_path << std::endl;
  std::ofstream out_file(db_path, std::ios::binary);
  if (!out_file) {
    std::cerr << "Error opening file: " << db_path << std::endl;
    return;
  }

  const auto num_nodes = features.num_nodes();
  const auto feature_dim = features.feature_dim();
  if (num_nodes == 0 || feature_dim == 0) {
    std::cerr << "Got empty node features\n";
    return;
  }

  out_file.write(reinterpret_cast<const char *>(&num_nodes), sizeof(num_nodes));
  out_file.write(reinterpret_cast<const char *>(&feature_dim),
                 sizeof(feature_dim));
  for (std::size_t node = 0; node < num_nodes; ++node) {
    const auto feat = features.get_feature(node);
    if (feat.size() != feature_dim) {
      std::cerr << "Row has wrong feature size\n";
      return;
    }
    out_file.write(reinterpret_cast<const char *>(feat.data()),
                   feature_dim * sizeof(feat[0]));
  }
}

inline auto read_file(const std::string &db_path) -> InMemoryOldFeatureStore {
  std::cout << "Reading graph at db_path: " << db_path << std::endl;
  std::ifstream in_file(db_path, std::ios::binary);
  if (!in_file) {
    std::cerr << "Error opening file: " << db_path << std::endl;
    return InMemoryOldFeatureStore({});
  }

  std::size_t num_nodes = 0;
  std::size_t feature_dim = 0;
  in_file.read(reinterpret_cast<char *>(&num_nodes), sizeof(num_nodes));
  in_file.read(reinterpret_cast<char *>(&feature_dim), sizeof(feature_dim));

  if (num_nodes == 0 || feature_dim == 0) {
    std::cerr << "Got empty node features\n";
    return InMemoryOldFeatureStore({});
  }

  std::vector<std::vector<float>> features(num_nodes,
                                           std::vector<float>(feature_dim));
  for (std::size_t i = 0; i < num_nodes; ++i) {
    in_file.read(reinterpret_cast<char *>(features[i].data()),
                 feature_dim * sizeof(float));
  }
  return InMemoryOldFeatureStore(features);
}

}  // namespace detail

inline auto init(const Config &config) -> Context {
  Context ctx = {.config = config};
  return ctx;
}

inline auto build(const Context &ctx,
                  std::span<const std::pair<NodeID, NodeID>> edges,
                  const OldFeatureStore &features) -> void {
  std::cout << "Number of edges: " << edges.size() << std::endl;
  std::cout << "Number of nodes: " << features.num_nodes() << std::endl;
  std::cout << "Feature Dim: " << features.feature_dim() << std::endl;
  detail::write_file(ctx.config.db_path, features);
}

inline auto gather(const Context &ctx, const std::vector<NodeID> &nodes)
    -> std::vector<std::vector<float>> {
  const auto features = detail::read_file(ctx.config.db_path);
  const std::size_t batch_size = nodes.size();
  const std::size_t feature_dim = features.feature_dim();

  std::vector<std::vector<float>> batch_features(
      batch_size, std::vector<float>(feature_dim));
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto feat = features.get_feature(nodes[i]);
    batch_features[i].assign(feat.begin(), feat.end());
  }
  return batch_features;
}

}  // namespace ggb
