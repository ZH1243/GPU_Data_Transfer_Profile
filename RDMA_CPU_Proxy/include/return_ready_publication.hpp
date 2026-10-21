#pragma once

#include <cstdint>

namespace rdma_proxy {

// Calls are serialized by ReturnGroupProgress. Separate pinned CPU and HBM
// flags; the nonblocking publication stream never waits on a compute stream.
class ReturnReadyPublication {
public:
    ReturnReadyPublication(int device, bool mock);
    ~ReturnReadyPublication();
    ReturnReadyPublication(const ReturnReadyPublication&) = delete;
    ReturnReadyPublication& operator=(const ReturnReadyPublication&) = delete;

    void reset(); // Publish zero and wait before allowing next-iteration writes.
    void publish(int32_t prefix); // Host asynchronous, default memory barrier.
    bool complete() const;
    void drain() const;
    int32_t host_value() const;
    uintptr_t host_address() const { return reinterpret_cast<uintptr_t>(host_); }
    uintptr_t device_address() const { return reinterpret_cast<uintptr_t>(device_flag_); }

private:
    void release() noexcept;
    int device_;
    bool mock_;
    int32_t* host_{nullptr};
    int32_t* device_flag_{nullptr};
    void* stream_{nullptr};
};
} // namespace rdma_proxy
