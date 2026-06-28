#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rdma_proxy {

namespace {

__global__ void copy_bytes_kernel(std::uint8_t* dst, const std::uint8_t* src, std::size_t bytes) {
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = tid; i < bytes; i += stride) {
        dst[i] = src[i];
    }
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

void launch_copy_tokens_kernel(void* dst, const void* src, std::size_t bytes) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((bytes + threads - 1) / threads);
    copy_bytes_kernel<<<std::max(1, blocks), threads>>>(
        static_cast<std::uint8_t*>(dst),
        static_cast<const std::uint8_t*>(src),
        bytes);
    check_cuda(cudaGetLastError(), "copy_bytes_kernel launch");
}

}  // namespace rdma_proxy
