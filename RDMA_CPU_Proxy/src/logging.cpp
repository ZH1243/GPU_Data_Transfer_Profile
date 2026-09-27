#include "logging.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>

namespace rdma_proxy {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_.store(level, std::memory_order_relaxed);
}

LogLevel Logger::level() const {
    return level_.load(std::memory_order_relaxed);
}

void Logger::log(LogLevel level, const std::string& message) {
    if (!enabled(level)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled(level)) {
        return;
    }

    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char timestamp[32]{};
    std::strftime(timestamp, sizeof(timestamp), "%F %T", &tm);
    std::cerr << "[" << timestamp << "] [" << to_string(level) << "] " << message << '\n';
}

const char* to_string(LogLevel level) {
    switch (level) {
        case LogLevel::kDebug:
            return "debug";
        case LogLevel::kInfo:
            return "info";
        case LogLevel::kWarn:
            return "warn";
        case LogLevel::kError:
            return "error";
    }
    return "unknown";
}

LogLevel log_level_from_string(const std::string& value) {
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "debug") return LogLevel::kDebug;
    if (v == "info") return LogLevel::kInfo;
    if (v == "warn" || v == "warning") return LogLevel::kWarn;
    if (v == "error") return LogLevel::kError;
    return LogLevel::kInfo;
}

}  // namespace rdma_proxy
