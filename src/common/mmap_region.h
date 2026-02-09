#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "logging.h"

namespace ggb::detail {

class MmapRegion {
 public:
  explicit MmapRegion(std::string path) : path_(std::move(path)) {
    auto fd = open(path_.c_str(), O_RDONLY);
    if (fd == -1) {
      GGB_LOG_ERROR("Failed to open file: {}", path_);
      throw std::runtime_error("MmapRegion: open failed");
    }

    struct stat st;
    if (fstat(fd, &st) == -1) {
      close(fd);
      GGB_LOG_ERROR("Fstat failed on: {}", path_);
      throw std::runtime_error("MmapRegion: fstat failed");
    }

    size_ = st.st_size;
    if (size_ == 0) {
      close(fd);
      GGB_LOG_ERROR("Attempted to mmap an empty file: {}", path_);
      throw std::runtime_error("MmapRegion: empty file");
    }

    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (!mmap_data_ptr_is_valid(data_)) {
      GGB_LOG_ERROR("MMap failed for {}, errno: {}", path_, errno);
      throw std::runtime_error("MmapRegion: mmap failed");
    }

    GGB_LOG_DEBUG("Mapped {} ({:.2f} GB)", path,
                  static_cast<double>(size_) / (1024 * 1024 * 1024));
  }

  ~MmapRegion() {
    if (mmap_data_ptr_is_valid(data_)) {
      try {
        munmap(data_, size_);
        GGB_LOG_DEBUG("Unmapped {}", path_);
      } catch (...) {
        GGB_LOG_ERROR("Unmap failed for {}", path_);
      }
    }
  }

  MmapRegion(const MmapRegion&) = delete;
  auto operator=(const MmapRegion&) -> MmapRegion& = delete;

  MmapRegion(MmapRegion&& other) noexcept { *this = std::move(other); }

  auto operator=(MmapRegion&& other) noexcept -> MmapRegion& {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(path_, other.path_);
    return *this;
  }

  auto advise(int advice) const -> void {
    if (mmap_data_ptr_is_valid(data_)) {
      madvise(data_, size_, advice);
    }
  }

  [[nodiscard]] auto data() const -> void* { return data_; }
  [[nodiscard]] auto size() const -> std::size_t { return size_; }

 private:
  std::string path_;
  std::size_t size_;
  void* data_ = nullptr;

  static auto mmap_data_ptr_is_valid(void* ptr) -> bool {
    return ptr != nullptr && ptr != MAP_FAILED;
  }
};
}  // namespace ggb::detail
