#include "cpu_affinity.hpp"

#include "logging.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace rdma_proxy {
namespace {

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream ss(value);
    std::string part;
    while (std::getline(ss, part, delimiter)) {
        parts.push_back(trim(part));
    }
    return parts;
}

bool is_cpu_list_token(const std::string& token) {
    if (token.empty()) return false;
    bool has_digit = false;
    for (const unsigned char c : token) {
        if (std::isdigit(c) != 0) {
            has_digit = true;
            continue;
        }
        if (c == '-' || c == ',') continue;
        return false;
    }
    return has_digit;
}

std::vector<int> parse_cpu_list(const std::string& cpu_list) {
    std::vector<int> cpus;
    for (const auto& part : split(cpu_list, ',')) {
        if (part.empty()) throw std::runtime_error("empty CPU affinity element in: " + cpu_list);
        const auto dash = part.find('-');
        if (dash == std::string::npos) {
            cpus.push_back(std::stoi(part));
            continue;
        }
        const int begin = std::stoi(part.substr(0, dash));
        const int end = std::stoi(part.substr(dash + 1));
        if (begin > end) throw std::runtime_error("invalid CPU affinity range: " + part);
        for (int cpu = begin; cpu <= end; ++cpu) {
            cpus.push_back(cpu);
        }
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    if (cpus.empty()) throw std::runtime_error("CPU affinity list is empty");
    return cpus;
}

std::string join_cpu_list(const std::vector<int>& cpus) {
    std::ostringstream out;
    for (std::size_t i = 0; i < cpus.size(); ++i) {
        if (i != 0) out << ',';
        out << cpus[i];
    }
    return out.str();
}

std::string resolve_auto_cpu_affinity(int cuda_device_id) {
#if defined(__linux__)
    FILE* pipe = popen("nvidia-smi topo -m 2>/dev/null", "r");
    if (!pipe) {
        throw std::runtime_error("failed to run nvidia-smi topo -m for cpu_affinity=auto");
    }

    const std::string gpu_prefix = "GPU" + std::to_string(cuda_device_id);
    char buffer[8192];
    std::string matching_line;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        const std::string line(buffer);
        std::istringstream tokens(line);
        std::string first;
        tokens >> first;
        if (first == gpu_prefix) {
            matching_line = line;
            break;
        }
    }

    const int status = pclose(pipe);
    if (matching_line.empty()) {
        throw std::runtime_error("nvidia-smi topo -m did not contain " + gpu_prefix);
    }
    if (status != 0) {
        RDMA_PROXY_LOG_WARN("nvidia-smi topo -m exited with status ", status,
                            "; using parsed ", gpu_prefix, " affinity anyway");
    }

    std::istringstream tokens(matching_line);
    std::vector<std::string> parts;
    std::string token;
    while (tokens >> token) {
        parts.push_back(token);
    }
    if (parts.size() >= 4 && is_cpu_list_token(parts[parts.size() - 3])) {
        return parts[parts.size() - 3];
    }
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        if (is_cpu_list_token(*it) && (it->find('-') != std::string::npos || it->find(',') != std::string::npos)) {
            return *it;
        }
    }
    throw std::runtime_error("could not parse CPU affinity from nvidia-smi topo row: " + matching_line);
#else
    (void)cuda_device_id;
    throw std::runtime_error("cpu_affinity=auto requires Linux and nvidia-smi");
#endif
}

std::string resolve_cpu_affinity(const ProxyConfig& config) {
    std::string spec = trim(config.cpu_affinity);
    std::transform(spec.begin(), spec.end(), spec.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (spec.empty() || spec == "none" || spec == "off" || spec == "disabled") return {};
    if (spec == "auto") return resolve_auto_cpu_affinity(config.cuda_device_id);
    return config.cpu_affinity;
}

}  // namespace

void apply_cpu_affinity(const ProxyConfig& config) {
    const auto affinity = resolve_cpu_affinity(config);
    if (affinity.empty()) {
        RDMA_PROXY_LOG_INFO("CPU affinity binding disabled");
        return;
    }

    const auto cpus = parse_cpu_list(affinity);
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            throw std::runtime_error("CPU id " + std::to_string(cpu) +
                                     " is outside supported CPU_SETSIZE " + std::to_string(CPU_SETSIZE));
        }
        CPU_SET(cpu, &set);
    }

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        throw std::runtime_error("sched_setaffinity failed for CPU list " + affinity + ": " +
                                 std::strerror(errno));
    }
    RDMA_PROXY_LOG_INFO("bound proxy process to CPU affinity ", affinity,
                        " (", cpus.size(), " CPU(s): ", join_cpu_list(cpus), ")");
#else
    throw std::runtime_error("cpu_affinity=" + affinity + " requires Linux sched_setaffinity");
#endif
}

}  // namespace rdma_proxy
