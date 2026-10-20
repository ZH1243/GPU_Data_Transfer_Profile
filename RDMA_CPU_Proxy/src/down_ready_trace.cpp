#include "down_ready_trace.hpp"
#include <stdexcept>
#include <string>
#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace rdma_proxy {

DownReadyTrace::~DownReadyTrace() {
    if (!host_) return;
    if (mock_) {
        delete host_;
    } else {
#if RDMA_PROXY_HAVE_CUDA
        cudaSetDevice(device_);
        // Protect the mapped allocation even during exceptional teardown.
        cudaDeviceSynchronize();
        cudaFreeHost(host_);
#endif
    }
}

uintptr_t DownReadyTrace::configure(int device, int groups, int timeout_ms, bool mock) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (host_ || groups <= 0 || timeout_ms <= 0)
        throw std::runtime_error("down-ready trace requires one configuration and positive groups/timeout");
    void* device_pointer = nullptr;
    if (mock) {
        host_ = new int32_t(0);
        device_pointer = host_;
    } else {
#if RDMA_PROXY_HAVE_CUDA
        auto check = [](cudaError_t status) {
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("down-ready trace: ") + cudaGetErrorString(status));
        };
        check(cudaSetDevice(device));
        check(cudaHostAlloc(reinterpret_cast<void**>(&host_), sizeof(*host_), cudaHostAllocMapped));
        const auto status = cudaHostGetDevicePointer(&device_pointer, host_, 0);
        if (status != cudaSuccess) {
            cudaFreeHost(host_);
            host_ = nullptr;
            check(status);
        }
#else
        throw std::runtime_error("down-ready trace requires CUDA");
#endif
    }
    mock_ = mock;
    device_ = device;
    groups_ = groups;
    timeout_ms_ = timeout_ms;
    observations_.reserve(groups);
    __atomic_store_n(host_, 0, __ATOMIC_RELEASE);
    return reinterpret_cast<uintptr_t>(device_pointer);
}

void DownReadyTrace::reset(uint64_t iteration) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!host_) return;
    if (active_.load()) throw std::runtime_error("previous down-ready trace is still active");
    if (returns_) returns_->reset();
    __atomic_store_n(host_, 0, __ATOMIC_RELEASE);
    observations_.clear();
    iteration_ = iteration;
    previous_ = 0;
    complete_ = cancelled_ = false;
    armed_ = true;
    start_ = std::chrono::steady_clock::now();
    active_.store(true);
}

void DownReadyTrace::poll() {
    if (!active_.load()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_.load()) return;
    const int ready = __atomic_load_n(host_, __ATOMIC_ACQUIRE);
    const auto now = std::chrono::steady_clock::now();
    if (ready < previous_ || ready > groups_)
        throw std::runtime_error("invalid down-ready prefix");
    if (ready > previous_) {
        observations_.emplace_back(
            ready, std::chrono::duration<double, std::micro>(now - start_).count());
        previous_ = ready;
    }
    const bool returned = !returns_ || returns_->poll(ready);
    if (ready == groups_ && returned) {
        complete_ = true;
        active_.store(false);
    } else if (now - start_ > std::chrono::milliseconds(timeout_ms_)) {
        throw std::runtime_error("timed out polling down-GEMM readiness or return completion");
    }
}

void DownReadyTrace::configure_returns(
    const RdmaProxyReturnCopy* copies, std::size_t count, const int64_t* rows,
    std::size_t row_count, std::size_t width, std::size_t tile,
    std::size_t group_tiles, std::size_t element_bytes, bool per_group, uintptr_t down_done) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!host_ || armed_ || returns_) throw std::runtime_error("invalid return configuration state");
    returns_ = std::make_unique<ReturnCopies>(
        device_, mock_, groups_, copies, count, rows, row_count, width,
        tile, group_tiles, element_bytes, per_group, down_done);
}

void DownReadyTrace::arm_returns(uint64_t iteration) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!returns_ || !armed_ || cancelled_ || iteration != iteration_ || !active_.load())
        throw std::runtime_error("invalid return arm state/iteration");
    returns_->arm();
}

void DownReadyTrace::drain_returns() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (returns_) returns_->drain();
}

void DownReadyTrace::cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_ = true;
    active_.store(false);
}

bool DownReadyTrace::read(
    uint64_t iteration, std::vector<std::pair<int32_t, double>>& observations) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!armed_ || iteration != iteration_) throw std::runtime_error("down-ready trace iteration mismatch");
    if (cancelled_) throw std::runtime_error("down-ready trace cancelled");
    if (!complete_) return false;
    observations = observations_;
    return true;
}
}  // namespace rdma_proxy
