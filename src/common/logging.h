#pragma once

#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <string_view>

namespace ggb::detail {

enum class LogLevel {
  DEBUG,
  INFO,
  WARN,
  ERROR,
};

inline void log_impl(LogLevel level, std::string_view file, int line,
                     std::string_view msg) {
  std::string_view prefix;
  auto show_trigger_loc = true;
  switch (level) {
    case LogLevel::DEBUG:
      prefix = "[DEBUG]";
      break;
    case LogLevel::WARN:
      prefix = "[WARN ]";
      break;
    case LogLevel::ERROR:
      prefix = "[ERR  ]";
      break;
    case LogLevel::INFO:
    default:  // Default to INFO
      prefix = "[INFO ]";
      show_trigger_loc = false;
      break;
  }

  if (show_trigger_loc) {
    auto short_file = std::filesystem::path(file).filename().string();
    std::cout << std::format("{} [{}:{}] {}\n", prefix, short_file, line, msg);
  } else {
    std::cout << std::format("{} {}\n", prefix, msg);
  }
}

}  // namespace ggb::detail

#define GGB_LOG_DEBUG(msg, ...)                                           \
  ggb::detail::log_impl(ggb::detail::LogLevel::DEBUG, __FILE__, __LINE__, \
                        std::format(msg __VA_OPT__(, ) __VA_ARGS__))

#define GGB_LOG_INFO(msg, ...)                                           \
  ggb::detail::log_impl(ggb::detail::LogLevel::INFO, __FILE__, __LINE__, \
                        std::format(msg __VA_OPT__(, ) __VA_ARGS__))

#define GGB_LOG_WARN(msg, ...)                                           \
  ggb::detail::log_impl(ggb::detail::LogLevel::WARN, __FILE__, __LINE__, \
                        std::format(msg __VA_OPT__(, ) __VA_ARGS__))

#define GGB_LOG_ERROR(msg, ...)                                           \
  ggb::detail::log_impl(ggb::detail::LogLevel::ERROR, __FILE__, __LINE__, \
                        std::format(msg __VA_OPT__(, ) __VA_ARGS__))
