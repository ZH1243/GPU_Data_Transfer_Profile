#include "cuda_buffers.hpp"

#include "logging.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
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

uint16_t float_to_bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

uint16_t float_to_fp16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U))) {
            ++half_mantissa;
        }
        return static_cast<uint16_t>(sign | half_mantissa);
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U))) {
        ++half_mantissa;
        if (half_mantissa == 0x400U) {
            half_mantissa = 0;
            if (exponent + 1 >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent) << 10) | half_mantissa);
}

std::vector<uint16_t> make_deterministic_weights(std::size_t elements, DataType dtype) {
    std::vector<uint16_t> weights(elements);
    uint64_t state = 0x6a09e667f3bcc909ULL;
    for (std::size_t i = 0; i < elements; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const auto bucket = static_cast<int>((state * 0x2545f4914f6cdd1dULL) % 33ULL) - 16;
        const float value = static_cast<float>(bucket) / 64.0F;
        weights[i] = dtype == DataType::kBF16 ? float_to_bf16_bits(value) : float_to_fp16_bits(value);
    }
    return weights;
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
    for (auto& entry : nvlink_recv_buffers_) {
        free_buffer(entry.recv);
    }
    for (auto& entry : nvlink_computation_output_buffers_) {
        free_buffer(entry.output);
    }
    free_buffer(nvlink_computation_weight_buffer_);
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

    nvlink_recv_buffers_.clear();
    if (config_.nvlink_forwarding_enabled) {
        const auto forwarding_bytes = nvlink_receive_buffer_bytes();
        nvlink_recv_buffers_.reserve(static_cast<std::size_t>(config_.num_gpus_per_node - 1));
        for (int source_gpu = 0; source_gpu < config_.num_gpus_per_node; ++source_gpu) {
            if (source_gpu == config_.local_gpu_index) continue;
            NvlinkReceiveBuffer entry;
            entry.source_gpu_index = source_gpu;
            allocate_buffer(entry.recv, forwarding_bytes);
            nvlink_recv_buffers_.push_back(entry);
            RDMA_PROXY_LOG_INFO("allocated NVLink receive buffer local_gpu=", config_.local_gpu_index,
                                " source_gpu=", source_gpu,
                                " bytes=", forwarding_bytes);
        }
    }

    nvlink_computation_output_buffers_.clear();
    if (config_.nvlink_forward_computation_enabled) {
        const auto output_bytes = nvlink_computation_output_buffer_bytes();
        nvlink_computation_output_buffers_.reserve(nvlink_recv_buffers_.size());
        for (const auto& recv : nvlink_recv_buffers_) {
            NvlinkComputationOutputBuffer entry;
            entry.source_gpu_index = recv.source_gpu_index;
            allocate_buffer(entry.output, output_bytes);
            nvlink_computation_output_buffers_.push_back(entry);
            RDMA_PROXY_LOG_INFO("allocated NVLink computation output buffer local_gpu=",
                                config_.local_gpu_index,
                                " source_gpu=", recv.source_gpu_index,
                                " bytes=", output_bytes);
        }

        const auto weight_elements =
            config_.token_dimension * config_.nvlink_forward_computation_output_dim;
        const auto weights = make_deterministic_weights(weight_elements, config_.dtype);
        allocate_buffer(nvlink_computation_weight_buffer_, weights.size() * sizeof(uint16_t));
        if (config_.mock_mode) {
            std::memcpy(nvlink_computation_weight_buffer_.ptr, weights.data(),
                        nvlink_computation_weight_buffer_.bytes);
        } else {
#if RDMA_PROXY_HAVE_CUDA
            check_cuda(cudaMemcpy(
                           nvlink_computation_weight_buffer_.ptr,
                           weights.data(),
                           nvlink_computation_weight_buffer_.bytes,
                           cudaMemcpyHostToDevice),
                       "cudaMemcpy H2D deterministic NVLink computation weights");
#else
            throw std::runtime_error("CUDA weight initialization requested but CUDA support was not built");
#endif
        }
        RDMA_PROXY_LOG_INFO("initialized deterministic NVLink computation weights K=",
                            config_.token_dimension,
                            " N=", config_.nvlink_forward_computation_output_dim,
                            " bytes=", nvlink_computation_weight_buffer_.bytes);
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

bool CudaBuffers::validate_recv_pattern(
    int peer_rank,
    int source_rank,
    int destination_rank,
    uint64_t iteration,
    const std::vector<ChunkDescriptor>& chunks,
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

    const auto token_bytes = config_.token_dimension * dtype_size(config_.dtype);
    for (const auto& chunk : chunks) {
        if (chunk.source_token_indices.empty()) {
            throw std::runtime_error("discontinuous receive validation requires chunk token indices");
        }
        for (std::size_t ordinal = 0; ordinal < chunk.source_token_indices.size(); ++ordinal) {
            const auto source_token = chunk.source_token_indices[ordinal];
            const auto dst_offset = chunk.dst_offset_bytes + ordinal * token_bytes;
            const auto src_offset = source_token * token_bytes;
            if (dst_offset + token_bytes > actual.size()) {
                throw std::runtime_error("discontinuous receive validation chunk exceeds receive buffer");
            }
            for (std::size_t byte = 0; byte < token_bytes; ++byte) {
                const auto offset = dst_offset + byte;
                const auto expected = test_pattern_byte(
                    source_rank, destination_rank, config_.local_gpu_index, iteration, src_offset + byte);
                if (actual[offset] != expected) {
                    if (error) {
                        std::ostringstream out;
                        out << "peer=" << peer_rank
                            << " offset=" << offset
                            << " source_token=" << source_token
                            << " expected=0x" << std::hex << static_cast<unsigned>(expected)
                            << " actual=0x" << static_cast<unsigned>(actual[offset]);
                        *error = out.str();
                    }
                    return false;
                }
            }
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

const NvlinkReceiveBuffer& CudaBuffers::nvlink_receive_buffer_for_source(int source_gpu_index) const {
    auto it = std::find_if(nvlink_recv_buffers_.begin(), nvlink_recv_buffers_.end(), [&](const auto& entry) {
        return entry.source_gpu_index == source_gpu_index;
    });
    if (it == nvlink_recv_buffers_.end()) throw std::runtime_error("unknown NVLink source GPU");
    return *it;
}

const NvlinkComputationOutputBuffer& CudaBuffers::nvlink_computation_output_buffer_for_source(
    int source_gpu_index) const {
    auto it = std::find_if(
        nvlink_computation_output_buffers_.begin(),
        nvlink_computation_output_buffers_.end(),
        [&](const auto& entry) { return entry.source_gpu_index == source_gpu_index; });
    if (it == nvlink_computation_output_buffers_.end()) {
        throw std::runtime_error("unknown NVLink computation output source GPU");
    }
    return *it;
}

std::size_t CudaBuffers::token_buffer_bytes() const {
    return config_.num_tokens * config_.token_dimension * dtype_size(config_.dtype);
}

std::size_t CudaBuffers::nvlink_receive_buffer_bytes() const {
    return token_buffer_bytes() * config_.peers.size();
}

std::size_t CudaBuffers::nvlink_computation_output_buffer_bytes() const {
    return config_.num_tokens * config_.peers.size() *
        config_.nvlink_forward_computation_output_dim * dtype_size(config_.dtype);
}

std::size_t CudaBuffers::nvlink_computation_weight_buffer_bytes() const {
    return config_.token_dimension * config_.nvlink_forward_computation_output_dim * dtype_size(config_.dtype);
}

void CudaBuffers::clear_nvlink_computation_outputs(void* stream) {
    if (!config_.nvlink_forward_computation_enabled) return;
    for (auto& entry : nvlink_computation_output_buffers_) {
        if (config_.mock_mode) {
            std::memset(entry.output.ptr, 0, entry.output.bytes);
            continue;
        }
#if RDMA_PROXY_HAVE_CUDA
        if (stream) {
            check_cuda(cudaMemsetAsync(
                           entry.output.ptr,
                           0,
                           entry.output.bytes,
                           reinterpret_cast<cudaStream_t>(stream)),
                       "cudaMemsetAsync NVLink computation output");
        } else {
            check_cuda(cudaMemset(entry.output.ptr, 0, entry.output.bytes),
                       "cudaMemset NVLink computation output");
        }
#else
        (void)stream;
        throw std::runtime_error("CUDA output clear requested but CUDA support was not built");
#endif
    }
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

void* create_cuda_stream(int cuda_device_id, bool nonblocking, bool mock_mode) {
    if (mock_mode) return nullptr;
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaSetDevice(cuda_device_id), "cudaSetDevice before stream create");
    cudaStream_t stream = nullptr;
    const unsigned flags = nonblocking ? cudaStreamNonBlocking : cudaStreamDefault;
    check_cuda(cudaStreamCreateWithFlags(&stream, flags), "cudaStreamCreateWithFlags");
    return reinterpret_cast<void*>(stream);
#else
    (void)cuda_device_id;
    (void)nonblocking;
    throw std::runtime_error("CUDA stream requested but CUDA support was not built");
#endif
}

void destroy_cuda_stream(void* stream, bool mock_mode) {
    if (mock_mode || !stream) return;
#if RDMA_PROXY_HAVE_CUDA
    const auto status = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
        RDMA_PROXY_LOG_WARN("cudaStreamDestroy failed during cleanup: ", cudaGetErrorString(status));
    }
#else
    (void)stream;
#endif
}

void synchronize_cuda_stream(void* stream, bool mock_mode) {
    if (mock_mode) return;
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
#else
    (void)stream;
    throw std::runtime_error("CUDA stream synchronization requested but CUDA support was not built");
#endif
}

void enable_cuda_peer_access(int cuda_device_id, int peer_cuda_device_id, bool mock_mode) {
    if (mock_mode || cuda_device_id == peer_cuda_device_id) return;
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaSetDevice(cuda_device_id), "cudaSetDevice before peer access enable");
    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, cuda_device_id, peer_cuda_device_id), "cudaDeviceCanAccessPeer");
    if (!can_access) {
        throw std::runtime_error("CUDA device " + std::to_string(cuda_device_id) +
                                 " cannot access peer CUDA device " + std::to_string(peer_cuda_device_id));
    }
    const auto status = cudaDeviceEnablePeerAccess(peer_cuda_device_id, 0);
    if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
        throw std::runtime_error(std::string("cudaDeviceEnablePeerAccess: ") + cudaGetErrorString(status));
    }
    if (status == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    }
#else
    (void)cuda_device_id;
    (void)peer_cuda_device_id;
    throw std::runtime_error("CUDA peer access requested but CUDA support was not built");
#endif
}

void launch_cuda_forward_copy_batch_async(
    const CudaForwardCopy& copy,
    void* stream,
    bool use_batch_api,
    bool mock_mode) {
    if (!copy.dst || !copy.src) throw std::runtime_error("NVLink forwarding copy has null pointer");
    if (copy.bytes == 0) throw std::runtime_error("NVLink forwarding copy has zero size");
    if (mock_mode) {
        std::memcpy(copy.dst, copy.src, copy.bytes);
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (use_batch_api) {
        void* dsts[] = {copy.dst};
        void* srcs[] = {const_cast<void*>(copy.src)};
        std::size_t sizes[] = {copy.bytes};
        cudaMemcpyAttributes attrs{};
        attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        std::size_t attrs_idxs[] = {0};
        std::size_t fail_idx = 0;
        check_cuda(cudaMemcpyBatchAsync(dsts, srcs, sizes, 1, &attrs, attrs_idxs, 1, &fail_idx, cuda_stream),
                   "cudaMemcpyBatchAsync");
    } else {
        check_cuda(cudaMemcpyAsync(copy.dst, copy.src, copy.bytes, cudaMemcpyDeviceToDevice, cuda_stream),
                   "cudaMemcpyAsync D2D forwarding");
    }
#else
    (void)copy;
    (void)stream;
    (void)use_batch_api;
    throw std::runtime_error("CUDA forwarding copy requested but CUDA support was not built");
#endif
}

void launch_cuda_forward_copy_batch_async(
    const std::vector<CudaForwardCopy>& copies,
    void* stream,
    bool use_batch_api,
    bool mock_mode) {
    if (copies.empty()) return;
    for (const auto& copy : copies) {
        if (!copy.dst || !copy.src) throw std::runtime_error("NVLink forwarding copy has null pointer");
        if (copy.bytes == 0) throw std::runtime_error("NVLink forwarding copy has zero size");
    }
    if (mock_mode) {
        for (const auto& copy : copies) {
            std::memcpy(copy.dst, copy.src, copy.bytes);
        }
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (use_batch_api) {
        std::vector<void*> dsts;
        std::vector<void*> srcs;
        std::vector<std::size_t> sizes;
        dsts.reserve(copies.size());
        srcs.reserve(copies.size());
        sizes.reserve(copies.size());
        for (const auto& copy : copies) {
            dsts.push_back(copy.dst);
            srcs.push_back(const_cast<void*>(copy.src));
            sizes.push_back(copy.bytes);
        }
        cudaMemcpyAttributes attrs{};
        attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        std::vector<std::size_t> attrs_idxs(copies.size(), 0);
        std::size_t fail_idx = 0;
        check_cuda(cudaMemcpyBatchAsync(
                       dsts.data(),
                       srcs.data(),
                       sizes.data(),
                       copies.size(),
                       &attrs,
                       attrs_idxs.data(),
                       1,
                       &fail_idx,
                       cuda_stream),
                   "cudaMemcpyBatchAsync");
    } else {
        for (const auto& copy : copies) {
            check_cuda(cudaMemcpyAsync(copy.dst, copy.src, copy.bytes, cudaMemcpyDeviceToDevice, cuda_stream),
                       "cudaMemcpyAsync D2D forwarding");
        }
    }
#else
    (void)copies;
    (void)stream;
    (void)use_batch_api;
    throw std::runtime_error("CUDA forwarding copy requested but CUDA support was not built");
#endif
}

std::string export_cuda_ipc_memory_handle(void* ptr, bool mock_mode) {
    if (!ptr) throw std::runtime_error("cannot export null CUDA IPC pointer");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    if (mock_mode) {
        out << std::setw(sizeof(uintptr_t) * 2) << reinterpret_cast<uintptr_t>(ptr);
        return out.str();
    }
#if RDMA_PROXY_HAVE_CUDA
    cudaIpcMemHandle_t handle{};
    check_cuda(cudaIpcGetMemHandle(&handle, ptr), "cudaIpcGetMemHandle");
    const auto* bytes = reinterpret_cast<const unsigned char*>(&handle);
    for (std::size_t i = 0; i < sizeof(handle); ++i) {
        out << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return out.str();
#else
    throw std::runtime_error("CUDA IPC export requested but CUDA support was not built");
#endif
}

void* open_cuda_ipc_memory_handle(const std::string& handle_hex, uint64_t mock_addr, bool mock_mode) {
    if (mock_mode) {
        return reinterpret_cast<void*>(static_cast<uintptr_t>(mock_addr));
    }
#if RDMA_PROXY_HAVE_CUDA
    cudaIpcMemHandle_t handle{};
    if (handle_hex.size() != sizeof(handle) * 2) {
        throw std::runtime_error("invalid CUDA IPC handle length");
    }
    auto* bytes = reinterpret_cast<unsigned char*>(&handle);
    for (std::size_t i = 0; i < sizeof(handle); ++i) {
        const auto byte_text = handle_hex.substr(i * 2, 2);
        bytes[i] = static_cast<unsigned char>(std::stoul(byte_text, nullptr, 16));
    }
    void* ptr = nullptr;
    check_cuda(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle");
    return ptr;
#else
    (void)handle_hex;
    (void)mock_addr;
    throw std::runtime_error("CUDA IPC open requested but CUDA support was not built");
#endif
}

void close_cuda_ipc_memory_handle(void* ptr, bool mock_mode) {
    if (mock_mode || !ptr) return;
#if RDMA_PROXY_HAVE_CUDA
    const auto status = cudaIpcCloseMemHandle(ptr);
    if (status != cudaSuccess) {
        RDMA_PROXY_LOG_WARN("cudaIpcCloseMemHandle failed during cleanup: ", cudaGetErrorString(status));
    }
#else
    (void)ptr;
#endif
}

}  // namespace rdma_proxy
