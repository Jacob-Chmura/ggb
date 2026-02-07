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

  friend auto operator<<(std::ostream &os, const FeatureStoreKey &key)
      -> std::ostream & {
    return os << "NodeID(" << key.NodeID << ")";
  }
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

namespace engine {

struct GGBConfig {
  const std::string db_path;
};

class GGBFeatureStore final : public FeatureStore {
 public:
  explicit GGBFeatureStore(
      GGBConfig cfg,
      std::unordered_map<FeatureStoreKey, std::size_t, FeatureStoreKeyHash>
          &&key_to_byte,
      std::optional<std::size_t> tensor_size)
      : cfg_(std::move(cfg)),
        key_to_byte_(std::move(key_to_byte)),
        tensor_size_(tensor_size) {}

  [[nodiscard]] auto get_num_keys() const -> std::size_t override {
    return key_to_byte_.size();
  }
  [[nodiscard]] auto get_tensor_size() const
      -> std::optional<std::size_t> override {
    return tensor_size_;
  }

  [[nodiscard]] auto get_multi_tensor_async(
      std::span<const FeatureStoreKey> keys) const
      -> std::future<std::vector<std::optional<FeatureStoreValue>>> override {
    std::vector<std::optional<FeatureStoreValue>> results;
    results.reserve(keys.size());

    std::ifstream file(cfg_.db_path, std::ios::binary);

    for (const auto &key : keys) {
      if (auto it = key_to_byte_.find(key); it != key_to_byte_.end()) {
        file.seekg(it->second);
        FeatureStoreValue val(tensor_size_.value());
        file.read(reinterpret_cast<char *>(val.data()),
                  tensor_size_.value() * sizeof(float));
        results.emplace_back(std::move(val));
      } else {
        results.emplace_back(std::nullopt);
      }
    }
    std::promise<std::vector<std::optional<FeatureStoreValue>>> promise;
    promise.set_value(std::move(results));
    return promise.get_future();
  }

 private:
  const GGBConfig cfg_;
  const std::unordered_map<FeatureStoreKey, std::size_t, FeatureStoreKeyHash>
      key_to_byte_;
  const std::optional<std::size_t> tensor_size_;
};

class GGBFeatureStoreBuilder final : public FeatureStoreBuilder {
 public:
  explicit GGBFeatureStoreBuilder(const GGBConfig &cfg)
      : cfg_(cfg), out_file_(cfg.db_path, std::ios::binary) {}

  auto put_tensor(const FeatureStoreKey &key, const FeatureStoreValue &tensor)
      -> bool override {
    if (!out_file_) {
      std::cerr << "Could not write to file: " << cfg_.db_path << "\n";
      return false;
    }

    if (key_to_byte_.contains(key)) {
      // TODO(kuba): Enable put overrides before build()
      std::cerr << key << " already present, skipping `put`\n";
      return false;
    }

    if (tensor_size_.has_value() && tensor.size() != tensor_size_.value()) {
      // TODO(kuba): Support various tensor sizes
      std::cerr << "Requested `put` on tensor of size '" << tensor.size()
                << "' but previously `put` a tensor of size '"
                << tensor_size_.value()
                << "'. Different tensor shapes are not yet supported\n";
      return false;
    }
    tensor_size_ = tensor.size();

    // do write
    key_to_byte_[key] = curr_offset_;
    auto bytes_to_write = tensor_size_.value() * sizeof(float);
    out_file_.write(reinterpret_cast<const char *>(tensor.data()),
                    bytes_to_write);
    curr_offset_ += bytes_to_write;
    return true;
  }

  auto put_tensor(const FeatureStoreKey &key, FeatureStoreValue &&tensor)
      -> bool override {
    // TODO(kuba): shoudl we do std::forward of somethign?
    return put_tensor(key, tensor);
  }

  [[nodiscard]] auto build(
      [[maybe_unused]] std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> override {
    out_file_.close();  // TODO(kuba): RAII
    return std::make_unique<GGBFeatureStore>(cfg_, std::move(key_to_byte_),
                                             std::move(tensor_size_));
  }

 private:
  const GGBConfig cfg_;
  std::ofstream out_file_;
  std::unordered_map<FeatureStoreKey, std::size_t, FeatureStoreKeyHash>
      key_to_byte_;
  std::optional<std::size_t> tensor_size_;
  std::size_t curr_offset_{0};
};

class GGBFeatureEngine : public FeatureEngine {
 public:
  explicit GGBFeatureEngine(GGBConfig cfg) : cfg_(std::move(cfg)) {}

  [[nodiscard]] auto create_builder()
      -> std::unique_ptr<FeatureStoreBuilder> override {
    return std::make_unique<GGBFeatureStoreBuilder>(cfg_);
  }

  [[nodiscard]] auto name() const -> std::string_view override { return name_; }

 private:
  static constexpr std::string_view name_ = "GGBFeatureEngine";
  const GGBConfig cfg_;
};

[[nodiscard]] inline auto create_ggb_engine(const GGBConfig &cfg)
    -> std::unique_ptr<FeatureEngine> {
  return std::make_unique<GGBFeatureEngine>(cfg);
}

}  // namespace engine
}  // namespace ggb
