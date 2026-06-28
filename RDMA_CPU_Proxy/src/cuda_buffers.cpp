#include "cuda_buffers.hpp"

#include "logging.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace rdma_proxy {
namespace {

#if RDMA_PROXY_HAVE_CUDA
void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}
#endif

uint8_t test_pattern_byte(int source_rank, int destination_rank, int gpu_index, uint64_t iteration, std::size_t offset) {
    uint64_t x = static_cast<uint64_t>(offset);
    x ^= (iteration + 1) * 0x9e3779b97f4a7c15ULL;
    x ^= (static_cast<uint64_t>(source_rank + 1) << 48);
    x ^= (static_cast<uint64_t>(destination_rank + 1) << 32);
    x ^= (static_cast<uint64_t>(gpu_index + 1) << 16);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return static_cast<uint8_t>(x & 0xffU);
}

std::vector<uint8_t> make_test_pattern(
    std::size_t bytes,
    int source_rank,
    int destination_rank,
    int gpu_index,
    uint64_t iteration) {
    std::vector<uint8_t> pattern(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        pattern[i] = test_pattern_byte(source_rank, destination_rank, gpu_index, iteration, i);
    }
    return pattern;
}

}  // namespace

#if RDMA_PROXY_HAVE_CUDA
void launch_copy_tokens_kernel(void* dst, const void* src, std::size_t bytes);
#endif

CudaBuffers::CudaBuffers(ProxyConfig config) : config_(std::move(config)) {}

CudaBuffers::~CudaBuffers() {
    for (auto& entry : buffers_) {
        free_buffer(entry.send);
        free_buffer(entry.recv);
    }
}

void CudaBuffers::initialize() {
    const auto bytes = token_buffer_bytes();
#if RDMA_PROXY_HAVE_CUDA
    if (!config_.mock_mode) {
        check_cuda(cudaSetDevice(config_.cuda_device_id), "cudaSetDevice");
    }
#else
    if (!config_.mock_mode) {
        throw std::runtime_error(
            "CUDA support was not built. Reconfigure with a CUDA compiler, or set mock_mode=true.");
    }
#endif

    buffers_.clear();
    buffers_.reserve(config_.peers.size());
    for (const auto& peer : config_.peers) {
        PeerGpuBuffers entry;
        entry.peer_rank = peer.node_rank;
        allocate_buffer(entry.send, bytes);
        allocate_buffer(entry.recv, bytes);
        buffers_.push_back(entry);
        RDMA_PROXY_LOG_INFO("allocated GPU buffers for peer ", peer.node_rank, " bytes=", bytes);
    }
}

void CudaBuffers::copy_tokens_to_send_buffer(int peer_rank, const void* src_device_or_host, std::size_t bytes) {
    auto& peer = buffers_for_peer(peer_rank);
    if (bytes > peer.send.bytes) {
        throw std::runtime_error("copy exceeds send buffer size");
    }
    launch_copy_tokens(peer.send.ptr, src_device_or_host, bytes, config_.mock_mode);
}

void CudaBuffers::fill_test_pattern(int peer_rank, int source_rank, int destination_rank, uint64_t iteration) {
    auto& peer = buffers_for_peer(peer_rank);
    const auto pattern = make_test_pattern(peer.send.bytes, source_rank, destination_rank, config_.local_gpu_index, iteration);
    if (config_.mock_mode) {
        std::memcpy(peer.send.ptr, pattern.data(), pattern.size());
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaMemcpy(peer.send.ptr, pattern.data(), pattern.size(), cudaMemcpyHostToDevice), "cudaMemcpy H2D test pattern");
#else
    throw std::runtime_error("CUDA test-pattern fill requested but CUDA support was not built");
#endif
}

bool CudaBuffers::validate_recv_pattern(
    int peer_rank,
    int source_rank,
    int destination_rank,
    uint64_t iteration,
    std::string* error) const {
    const auto& peer = buffers_for_peer(peer_rank);
    std::vector<uint8_t> actual(peer.recv.bytes);
    if (config_.mock_mode) {
        std::memcpy(actual.data(), peer.recv.ptr, actual.size());
    } else {
#if RDMA_PROXY_HAVE_CUDA
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before validation");
        check_cuda(cudaMemcpy(actual.data(), peer.recv.ptr, actual.size(), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H recv validation");
#else
        throw std::runtime_error("CUDA receive validation requested but CUDA support was not built");
#endif
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto expected = test_pattern_byte(source_rank, destination_rank, config_.local_gpu_index, iteration, i);
        if (actual[i] != expected) {
            if (error) {
                std::ostringstream out;
                out << "peer=" << peer_rank
                    << " offset=" << i
                    << " expected=0x" << std::hex << static_cast<unsigned>(expected)
                    << " actual=0x" << static_cast<unsigned>(actual[i]);
                *error = out.str();
            }
            return false;
        }
    }
    return true;
}

PeerGpuBuffers& CudaBuffers::buffers_for_peer(int peer_rank) {
    auto it = std::find_if(buffers_.begin(), buffers_.end(), [&](const auto& entry) {
        return entry.peer_rank == peer_rank;
    });
    if (it == buffers_.end()) throw std::runtime_error("unknown peer rank");
    return *it;
}

const PeerGpuBuffers& CudaBuffers::buffers_for_peer(int peer_rank) const {
    auto it = std::find_if(buffers_.begin(), buffers_.end(), [&](const auto& entry) {
        return entry.peer_rank == peer_rank;
    });
    if (it == buffers_.end()) throw std::runtime_error("unknown peer rank");
    return *it;
}

std::size_t CudaBuffers::token_buffer_bytes() const {
    return config_.num_tokens * config_.token_dimension * dtype_size(config_.dtype);
}

void CudaBuffers::allocate_buffer(GpuBuffer& buffer, std::size_t bytes) {
    buffer.bytes = bytes;
    buffer.is_mock_host_memory = config_.mock_mode;
    if (config_.mock_mode) {
        buffer.ptr = ::operator new(bytes);
        std::memset(buffer.ptr, 0, bytes);
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaMalloc(&buffer.ptr, bytes), "cudaMalloc");
    check_cuda(cudaMemset(buffer.ptr, 0, bytes), "cudaMemset");
#else
    throw std::runtime_error("CUDA allocation requested but CUDA support was not built");
#endif
}

void CudaBuffers::free_buffer(GpuBuffer& buffer) {
    if (!buffer.ptr) return;
    if (buffer.is_mock_host_memory) {
        ::operator delete(buffer.ptr);
    } else {
#if RDMA_PROXY_HAVE_CUDA
        const auto status = cudaFree(buffer.ptr);
        if (status != cudaSuccess) {
            RDMA_PROXY_LOG_WARN("cudaFree failed during cleanup: ", cudaGetErrorString(status));
        }
#endif
    }
    buffer.ptr = nullptr;
    buffer.bytes = 0;
}

void launch_copy_tokens(void* dst, const void* src, std::size_t bytes, bool mock_mode) {
    if (mock_mode) {
        std::memcpy(dst, src, bytes);
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    launch_copy_tokens_kernel(dst, src, bytes);
#else
    (void)dst;
    (void)src;
    (void)bytes;
    throw std::runtime_error("CUDA copy requested but CUDA support was not built");
#endif
}

}  // namespace rdma_proxy
