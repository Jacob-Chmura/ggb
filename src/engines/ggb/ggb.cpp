#include "ggb.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ggb::engine {

GGBFeatureStore::GGBFeatureStore(
    GGBConfig cfg, std::unordered_map<Key, std::size_t, KeyHash> &&key_to_byte,
    std::optional<std::size_t> tensor_size)
    : cfg_(std::move(cfg)),
      key_to_byte_(std::move(key_to_byte)),
      tensor_size_(tensor_size) {
  setup_mmap();
}

GGBFeatureStore::~GGBFeatureStore() { cleanup_mmap(); }

[[nodiscard]] auto GGBFeatureStore::name() const -> std::string_view {
  return name_;
}

[[nodiscard]] auto GGBFeatureStore::get_num_keys() const -> std::size_t {
  return key_to_byte_.size();
}

[[nodiscard]] auto GGBFeatureStore::get_tensor_size() const
    -> std::optional<std::size_t> {
  return tensor_size_;
}

[[nodiscard]] auto GGBFeatureStore::get_multi_tensor_async(
    std::span<const Key> keys) const
    -> std::future<std::vector<std::optional<Value>>> {
  std::vector<std::optional<Value>> results;
  results.reserve(keys.size());

  for (const auto &key : keys) {
    if (auto it = key_to_byte_.find(key); it != key_to_byte_.end()) {
      const float *start = mapped_data_ + (it->second / sizeof(float));
      results.emplace_back(Value(start, start + *tensor_size_));
    } else {
      results.emplace_back(std::nullopt);
    }
  }

  std::promise<std::vector<std::optional<Value>>> promise;
  promise.set_value(std::move(results));
  return promise.get_future();
}

void GGBFeatureStore::setup_mmap() {
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

  if (mapped_data_ == MAP_FAILED) {
    throw std::runtime_error("mmap failed");
  }
}

void GGBFeatureStore::cleanup_mmap() {
  if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
    munmap(const_cast<float *>(mapped_data_), file_size_);
    mapped_data_ = nullptr;
  }
}

GGBFeatureStoreBuilder::GGBFeatureStoreBuilder(const GGBConfig &cfg)
    : cfg_(cfg), out_file_(cfg.db_path, std::ios::binary) {}

auto GGBFeatureStoreBuilder::put_tensor(const Key &key, const Value &tensor)
    -> bool {
  if (!out_file_) {
    std::cerr << "Could not write to file: " << cfg_.db_path << "\n";
    return false;
  }
  if (tensor_size_.has_value() && tensor.size() != tensor_size_.value()) {
    std::cerr << "Requested `put` on tensor of size '" << tensor.size()
              << "' but previously `put` a tensor of size '"
              << tensor_size_.value()
              << "'. Different tensor shapes are not yet supported\n";
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

auto GGBFeatureStoreBuilder::put_tensor(const Key &key, Value &&tensor)
    -> bool {
  return put_tensor(key, static_cast<const Value &>(tensor));
}

[[nodiscard]] auto GGBFeatureStoreBuilder::build(
    [[maybe_unused]] std::optional<GraphTopology> graph)
    -> std::unique_ptr<FeatureStore> {
  out_file_.close();
  return std::make_unique<GGBFeatureStore>(cfg_, std::move(key_to_byte_),
                                           tensor_size_);
}

}  // namespace ggb::engine
