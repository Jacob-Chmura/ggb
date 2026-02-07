#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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

  [[nodiscard]] virtual auto name() const -> std::string_view = 0;
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
  virtual auto build(std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> = 0;
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
        tensor_size_(tensor_size) {
    // TODO(kuba): switch to std::format
    setup_mmap();
  }

  ~GGBFeatureStore() override { cleanup_mmap(); }

  [[nodiscard]] auto name() const -> std::string_view override { return name_; }

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
        // it->second is the byte offset, divide by sizeof(float) for the ptr
        const float *start = mapped_data_ + (it->second / sizeof(float));
        results.emplace_back(FeatureStoreValue(start, start + *tensor_size_));
      } else {
        results.emplace_back(std::nullopt);
      }
    }
    std::promise<std::vector<std::optional<FeatureStoreValue>>> promise;
    promise.set_value(std::move(results));
    return promise.get_future();
  }

 private:
  static constexpr std::string_view name_ = "GGBFeatureStore";
  const GGBConfig cfg_;
  const std::unordered_map<FeatureStoreKey, std::size_t, FeatureStoreKeyHash>
      key_to_byte_;
  const std::optional<std::size_t> tensor_size_;

  const float *mapped_data_ = nullptr;
  std::size_t file_size_{0};

  void cleanup_mmap() {
    if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
      munmap(const_cast<float *>(mapped_data_), file_size_);
      mapped_data_ = nullptr;
    }
  }

  void setup_mmap() {
    auto fd = open(cfg_.db_path.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Could not open " + cfg_.db_path);
    }

    struct stat st;
    fstat(fd, &st);
    file_size_ = st.st_size;
    mapped_data_ = static_cast<const float *>(
        mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd, 0));
    close(fd);
  }
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
    if (tensor_size_.has_value() && tensor.size() != tensor_size_.value()) {
      // TODO(kuba): Support various tensor sizes
      std::cerr << "Requested `put` on tensor of size '" << tensor.size()
                << "' but previously `put` a tensor of size '"
                << tensor_size_.value()
                << "'. Different tensor shapes are not yet supported\n";
      return false;
    }
    tensor_size_ = tensor.size();

    // Note: If key exists, we simply overwrite the offset in the map.
    key_to_byte_[key] = offset_;
    auto bytes_to_write = tensor_size_.value() * sizeof(float);
    out_file_.write(reinterpret_cast<const char *>(tensor.data()),
                    bytes_to_write);
    offset_ += bytes_to_write;
    return true;
  }

  auto put_tensor(const FeatureStoreKey &key, FeatureStoreValue &&tensor)
      -> bool override {
    return put_tensor(key, static_cast<const FeatureStoreValue &>(tensor));
  }

  [[nodiscard]] auto build(
      [[maybe_unused]] std::optional<GraphTopology> graph = std::nullopt)
      -> std::unique_ptr<FeatureStore> override {
    out_file_.close();
    return std::make_unique<GGBFeatureStore>(cfg_, std::move(key_to_byte_),
                                             tensor_size_);
  }

 private:
  const GGBConfig cfg_;
  std::ofstream out_file_;
  std::unordered_map<FeatureStoreKey, std::size_t, FeatureStoreKeyHash>
      key_to_byte_;
  std::optional<std::size_t> tensor_size_;
  std::size_t offset_{0};
};

[[nodiscard]] inline auto create_ggb_builder(const GGBConfig &cfg)
    -> std::unique_ptr<FeatureStoreBuilder> {
  return std::make_unique<GGBFeatureStoreBuilder>(cfg);
}

}  // namespace engine
}  // namespace ggb
