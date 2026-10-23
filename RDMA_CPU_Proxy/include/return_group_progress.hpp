#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <vector>
#include <utility>
#include "return_ready_publication.hpp"

namespace rdma_proxy {

// One writer (the forwarding submitter) consumes ordered per-peer messages.
// The mutex also serializes iteration reset and iteration-qualified snapshots.
class ReturnGroupProgress {
public:
    void configure(int peers, int self, int groups, bool publish_gpu = false,
                   int device = 0, bool mock = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (groups_ || peers < 2 || self < 0 || self >= peers || groups <= 0)
            throw std::runtime_error("invalid return-group progress configuration");
        if (publish_gpu) publication_ = std::make_unique<ReturnReadyPublication>(device, mock);
        groups_ = groups;
        peers_.resize(peers);
    }

    void reset(uint64_t iteration) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!groups_ || (active_ && (prefix_.load() != groups_ || iteration != iteration_ + 1)))
            throw std::runtime_error("previous return notifications have not drained or iteration is invalid");
        // Hold the progress lock through the GPU reset: an early peer message
        // cannot publish a new prefix ahead of zero or have it erased by zero.
        if (publication_) publication_->reset();
        iteration_ = iteration;
        std::fill(peers_.begin(), peers_.end(), 0);
        // Self advances through the same notifications as returned peer outputs.
        prefix_.store(0, std::memory_order_release);
        active_ = true;
    }

    // A faster peer can publish before our next iteration is prepared. Leave
    // that message queued until reset; never consume it into the old prefix.
    bool consume(int peer, uint64_t iteration, int group) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (peer < 0 || peer >= static_cast<int>(peers_.size()) ||
            group < 0 || group >= groups_)
            throw std::runtime_error("invalid return-group notification");
        if (!active_ || iteration > iteration_) return false;
        if (iteration != iteration_ || group != peers_[peer])
            throw std::runtime_error("stale, duplicate or out-of-order return-group notification");
        ++peers_[peer];
        const auto ready = *std::min_element(peers_.begin(), peers_.end());
        if (publication_ && ready > prefix_.load()) publication_->publish(ready);
        prefix_.store(ready, std::memory_order_release);
        return true;
    }

    int prefix(uint64_t iteration) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_ || iteration != iteration_)
            throw std::runtime_error("return-group progress iteration mismatch");
        return publication_ ? publication_->host_value() : prefix_.load(std::memory_order_acquire);
    }

    std::pair<uintptr_t, uintptr_t> publication_addresses() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!publication_) throw std::runtime_error("return-ready GPU publication is disabled or unconfigured");
        return {publication_->host_address(), publication_->device_address()};
    }

    bool publication_complete() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !publication_ || publication_->complete();
    }

    void shutdown_publication() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (publication_) publication_->drain();
        publication_.reset();
    }

private:
    mutable std::mutex mutex_;
    std::vector<int> peers_;
    int groups_{0};
    uint64_t iteration_{0};
    bool active_{false};
    std::unique_ptr<ReturnReadyPublication> publication_;
    // CPU fallback/progress bookkeeping. With GPU publication enabled, the
    // exposed host snapshot lives in publication_'s pinned allocation.
    std::atomic<int32_t> prefix_{0};
};
} // namespace rdma_proxy
