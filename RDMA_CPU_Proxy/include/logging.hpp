#pragma once

#include <mutex>
#include <sstream>
#include <string>

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
    void log(LogLevel level, const std::string& message);

private:
    Logger() = default;

    mutable std::mutex mutex_;
    LogLevel level_{LogLevel::kInfo};
};

const char* to_string(LogLevel level);
LogLevel log_level_from_string(const std::string& value);

template <typename... Args>
void log_message(LogLevel level, Args&&... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    Logger::instance().log(level, oss.str());
}

}  // namespace rdma_proxy

#define RDMA_PROXY_LOG_DEBUG(...) ::rdma_proxy::log_message(::rdma_proxy::LogLevel::kDebug, __VA_ARGS__)
#define RDMA_PROXY_LOG_INFO(...) ::rdma_proxy::log_message(::rdma_proxy::LogLevel::kInfo, __VA_ARGS__)
#define RDMA_PROXY_LOG_WARN(...) ::rdma_proxy::log_message(::rdma_proxy::LogLevel::kWarn, __VA_ARGS__)
#define RDMA_PROXY_LOG_ERROR(...) ::rdma_proxy::log_message(::rdma_proxy::LogLevel::kError, __VA_ARGS__)
