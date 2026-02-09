#include "flat_mmap.h"

#include <sys/mman.h>

#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/logging.h"

namespace ggb::engine {

FlatMmapFeatureStore::FlatMmapFeatureStore(
    FlatMmapConfig cfg,
    std::unordered_map<Key, std::size_t, KeyHash> &&key_to_byte,
    std::optional<std::size_t> tensor_size)
    : cfg_(std::move(cfg)),
      key_to_byte_(std::move(key_to_byte)),
      tensor_size_(tensor_size),
      mmap_(cfg_.db_path) {
  mmap_.advise(MADV_RANDOM);  // get some help from the kernel
}

[[nodiscard]] auto FlatMmapFeatureStore::name() const -> std::string_view {
  return name_;
}

[[nodiscard]] auto FlatMmapFeatureStore::get_num_keys() const -> std::size_t {
  return key_to_byte_.size();
}

[[nodiscard]] auto FlatMmapFeatureStore::get_tensor_size() const
    -> std::optional<std::size_t> {
  return tensor_size_;
}

[[nodiscard]] auto FlatMmapFeatureStore::get_multi_tensor_async(
    std::span<const Key> keys) const
    -> std::future<std::vector<std::optional<Value>>> {
  std::vector<std::optional<Value>> results;

  if (!tensor_size_.has_value()) {
    GGB_LOG_WARN("Empty tensor dimension found");
    results.assign(keys.size(), std::nullopt);
  } else {
    results.reserve(keys.size());
    const auto *const mapped_data = static_cast<const float *>(mmap_.data());

    for (const auto &key : keys) {
      if (auto it = key_to_byte_.find(key); it != key_to_byte_.end()) {
        const float *start = mapped_data + (it->second / sizeof(float));
        results.emplace_back(Value(start, start + tensor_size_.value()));
      } else {
        results.emplace_back(std::nullopt);
      }
    }
  }

  std::promise<std::vector<std::optional<Value>>> promise;
  promise.set_value(std::move(results));
  return promise.get_future();
}

FlatMmapFeatureStoreBuilder::FlatMmapFeatureStoreBuilder(
    const FlatMmapConfig &cfg)
    : cfg_(cfg), out_file_(cfg.db_path, std::ios::binary) {}

auto FlatMmapFeatureStoreBuilder::put_tensor_impl(const Key &key,
                                                  const Value &tensor) -> bool {
  if (!out_file_) {
    GGB_LOG_ERROR("Could not write to file: {}", cfg_.db_path);
    return false;
  }
  if (tensor_size_.has_value() && tensor.size() != tensor_size_.value()) {
    GGB_LOG_ERROR("Mismatched tensor size: got {}, expected {}", tensor.size(),
                  tensor_size_.value());
    return false;
  }
  tensor_size_ = tensor.size();

  key_to_byte_[key] = write_pos_;
  auto bytes_to_write = tensor_size_.value() * sizeof(float);
  out_file_.write(reinterpret_cast<const char *>(tensor.data()),
                  bytes_to_write);
  write_pos_ += bytes_to_write;
  return true;
}

auto FlatMmapFeatureStoreBuilder::put_tensor_impl(const Key &key,
                                                  Value &&tensor) -> bool {
  return put_tensor_impl(key, static_cast<const Value &>(tensor));
}

[[nodiscard]] auto FlatMmapFeatureStoreBuilder::build_impl(
    [[maybe_unused]] std::optional<GraphTopology> graph)
    -> std::unique_ptr<FeatureStore> {
  out_file_.close();
  GGB_LOG_INFO(
      "Building FlatMmapStore\n\tTotal Keys: {}\n\tFile Size: {:.3f} "
      "GB\n\tPath: {}",
      key_to_byte_.size(),
      static_cast<double>(write_pos_) / (1024 * 1024 * 1024), cfg_.db_path);
  return std::make_unique<FlatMmapFeatureStore>(cfg_, std::move(key_to_byte_),
                                                tensor_size_);
}

}  // namespace ggb::engine
