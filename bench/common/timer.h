#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace ggb::bench::perf {

class ScopedTimer {
 public:
  using Callback = std::function<void(std::uint64_t)>;

  explicit ScopedTimer(Callback cb)
      : cb_(std::move(cb)), start_(std::chrono::high_resolution_clock::now()) {}

  explicit ScopedTimer(std::string_view op_name = "Operation")
      : start_(std::chrono::high_resolution_clock::now()) {
    const std::string op_name_str{op_name};
    cb_ = [op_name_str](std::uint64_t us) {
      std::cout << std::format("[TIMER] {}: {} ms\n", op_name_str,
                               static_cast<double>(us) / 1000.0);
    };
  }

  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
    try {
      cb_(duration.count());
    } catch (...) {
      std::cerr << "Exception occured during ScopedTimer callback execution";
    }
  }

  ScopedTimer(const ScopedTimer&) = delete;
  auto operator=(const ScopedTimer&) -> ScopedTimer& = delete;

  ScopedTimer(ScopedTimer&&) = delete;
  auto operator=(ScopedTimer&&) -> ScopedTimer& = delete;

 private:
  Callback cb_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
}  // namespace ggb::bench::perf
