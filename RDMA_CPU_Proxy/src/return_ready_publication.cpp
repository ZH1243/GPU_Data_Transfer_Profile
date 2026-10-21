#include "return_ready_publication.hpp"
#include "cuda_buffers.hpp"

#include <stdexcept>
#include <string>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace rdma_proxy {
namespace {
#if RDMA_PROXY_HAVE_CUDA
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#endif
}

ReturnReadyPublication::ReturnReadyPublication(int device, bool mock)
    : device_(device), mock_(mock) {
    try {
        select_cuda_device_for_thread(device_, mock_);
        if (mock_) {
            host_ = new int32_t(0);
            device_flag_ = new int32_t(0);
        } else {
#if RDMA_PROXY_HAVE_CUDA
            check(cudaHostAlloc(reinterpret_cast<void**>(&host_), sizeof(int32_t), cudaHostAllocDefault),
                  "cudaHostAlloc return-ready flag");
            check(cudaMalloc(reinterpret_cast<void**>(&device_flag_), sizeof(int32_t)),
                  "cudaMalloc return-ready flag");
#else
            throw std::runtime_error("return-ready GPU publication requires CUDA");
#endif
        }
        stream_ = create_cuda_stream(device_, true, mock_);
        reset();
    } catch (...) {
        release();
        throw;
    }
}

ReturnReadyPublication::~ReturnReadyPublication() { release(); }

void ReturnReadyPublication::publish(int32_t prefix) {
    if (prefix < 0) throw std::runtime_error("negative return-ready prefix");
    select_cuda_device_for_thread(device_, mock_);
    if (mock_) {
        __atomic_store_n(device_flag_, prefix, __ATOMIC_RELEASE);
    } else {
#if RDMA_PROXY_HAVE_CUDA
        const auto status = cuStreamWriteValue32(
            reinterpret_cast<CUstream>(stream_), reinterpret_cast<CUdeviceptr>(device_flag_),
            static_cast<cuuint32_t>(prefix), CU_STREAM_WRITE_VALUE_DEFAULT);
        if (status != CUDA_SUCCESS) {
            const char* name = nullptr;
            const char* description = nullptr;
            cuGetErrorName(status, &name);
            cuGetErrorString(status, &description);
            throw std::runtime_error(std::string("cuStreamWriteValue32 return-ready flag: ") +
                (name ? name : "unknown CUDA error") + " " + (description ? description : ""));
        }
#endif
    }
    // The host snapshot denotes a successfully enqueued publication, not its
    // completion. cuStreamWriteValue32 captures prefix by value, so no pinned
    // source lifetime hazard exists when this integer changes again.
    __atomic_store_n(host_, prefix, __ATOMIC_RELEASE);
}

void ReturnReadyPublication::reset() {
    publish(0);
    drain();
}

int32_t ReturnReadyPublication::host_value() const {
    return __atomic_load_n(host_, __ATOMIC_ACQUIRE);
}

bool ReturnReadyPublication::complete() const {
    select_cuda_device_for_thread(device_, mock_);
    return query_cuda_stream(stream_, mock_);
}

void ReturnReadyPublication::drain() const {
    select_cuda_device_for_thread(device_, mock_);
    synchronize_cuda_stream(stream_, mock_);
}

void ReturnReadyPublication::release() noexcept {
    if (mock_) {
        delete host_;
        delete device_flag_;
    } else {
#if RDMA_PROXY_HAVE_CUDA
        cudaSetDevice(device_);
        if (stream_) cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
        if (device_flag_) cudaFree(device_flag_);
        if (host_) cudaFreeHost(host_);
#endif
    }
    // Normal shutdown drains explicitly and reports CUDA errors; this also
    // handles partial construction and exceptional teardown without throwing.
    try { destroy_cuda_stream(stream_, mock_); } catch (...) {}
    stream_ = nullptr;
    host_ = device_flag_ = nullptr;
}
} // namespace rdma_proxy
