#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace rdma_proxy {

// One mapped flag per proxy. Only the notification receiver calls poll().
// The embedder must finish GPU writes before destroying this object.
class DownReadyTrace {
public:
    ~DownReadyTrace();
    uintptr_t configure(int device, int groups, int timeout_ms, bool mock);
    void reset(uint64_t iteration);
    void poll();
    void cancel();
    bool read(uint64_t iteration, std::vector<std::pair<int32_t, double>>& observations);

private:
    std::mutex mutex_;
    std::atomic<bool> active_{false};
    int32_t* host_{nullptr};
    bool mock_{false};
    int device_{0}, groups_{0}, previous_{0}, timeout_ms_{0};
    uint64_t iteration_{0};
    bool armed_{false}, complete_{false}, cancelled_{false};
    std::chrono::steady_clock::time_point start_;
    std::vector<std::pair<int32_t, double>> observations_;
};
}  // namespace rdma_proxy
