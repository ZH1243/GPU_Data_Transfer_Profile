#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace rdma_proxy {

// One writer (the forwarding submitter) consumes ordered per-peer messages.
// The mutex also serializes iteration reset and iteration-qualified snapshots.
class ReturnGroupProgress {
public:
    void configure(int peers, int self, int groups) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (groups_ || peers < 2 || self < 0 || self >= peers || groups <= 0)
            throw std::runtime_error("invalid return-group progress configuration");
        groups_ = groups;
        self_ = self;
        peers_.resize(peers);
    }

    void reset(uint64_t iteration) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!groups_ || (active_ && (prefix_.load() != groups_ || iteration != iteration_ + 1)))
            throw std::runtime_error("previous return notifications have not drained or iteration is invalid");
        iteration_ = iteration;
        std::fill(peers_.begin(), peers_.end(), 0);
        peers_[self_] = groups_; // Locally retained outputs are not returned.
        prefix_.store(0, std::memory_order_release);
        active_ = true;
    }

    // A faster peer can publish before our next iteration is prepared. Leave
    // that message queued until reset; never consume it into the old prefix.
    bool consume(int peer, uint64_t iteration, int group) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (peer < 0 || peer >= static_cast<int>(peers_.size()) || peer == self_ ||
            group < 0 || group >= groups_)
            throw std::runtime_error("invalid return-group notification");
        if (!active_ || iteration > iteration_) return false;
        if (iteration != iteration_ || group != peers_[peer])
            throw std::runtime_error("stale, duplicate or out-of-order return-group notification");
        ++peers_[peer];
        prefix_.store(*std::min_element(peers_.begin(), peers_.end()), std::memory_order_release);
        return true;
    }

    int prefix(uint64_t iteration) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_ || iteration != iteration_)
            throw std::runtime_error("return-group progress iteration mismatch");
        return prefix_.load(std::memory_order_acquire);
    }

private:
    mutable std::mutex mutex_;
    std::vector<int> peers_;
    int groups_{0}, self_{-1};
    uint64_t iteration_{0};
    bool active_{false};
    // CPU DRAM only; not a GPU-readable flag.
    std::atomic<int32_t> prefix_{0};
};
} // namespace rdma_proxy
