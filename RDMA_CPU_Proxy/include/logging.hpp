#pragma once

#include <atomic>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace rdma_proxy {

enum class LogLevel {
    kDebug = 0,
    kInfo = 1,
    kWarn = 2,
    kError = 3,
};

class Logger {
public:
    static Logger& instance();

    void set_level(LogLevel level);
    LogLevel level() const;
    bool enabled(LogLevel level) const {
        return static_cast<int>(level) >= static_cast<int>(level_.load(std::memory_order_relaxed));
    }
    void log(LogLevel level, const std::string& message);

private:
    Logger() = default;

    mutable std::mutex mutex_;
    std::atomic<LogLevel> level_{LogLevel::kInfo};
};

const char* to_string(LogLevel level);
LogLevel log_level_from_string(const std::string& value);

template <typename... Args>
void log_message(LogLevel level, Args&&... args) {
    // Also protect direct callers from formatting suppressed messages.
    if (!Logger::instance().enabled(level)) return;
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    Logger::instance().log(level, oss.str());
}

}  // namespace rdma_proxy

// Check before evaluating arguments: disabled logs must not build strings,
// allocate formatting storage, or acquire the output mutex.
#define RDMA_PROXY_LOG_AT_LEVEL(level, ...) \
    do { \
        if (::rdma_proxy::Logger::instance().enabled(level)) { \
            ::rdma_proxy::log_message(level, __VA_ARGS__); \
        } \
    } while (false)
#define RDMA_PROXY_LOG_DEBUG(...) RDMA_PROXY_LOG_AT_LEVEL(::rdma_proxy::LogLevel::kDebug, __VA_ARGS__)
#define RDMA_PROXY_LOG_INFO(...) RDMA_PROXY_LOG_AT_LEVEL(::rdma_proxy::LogLevel::kInfo, __VA_ARGS__)
#define RDMA_PROXY_LOG_WARN(...) RDMA_PROXY_LOG_AT_LEVEL(::rdma_proxy::LogLevel::kWarn, __VA_ARGS__)
#define RDMA_PROXY_LOG_ERROR(...) RDMA_PROXY_LOG_AT_LEVEL(::rdma_proxy::LogLevel::kError, __VA_ARGS__)
