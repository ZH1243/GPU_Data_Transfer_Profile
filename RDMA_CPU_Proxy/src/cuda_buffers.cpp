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

constexpr std::size_t kNvlinkNotificationTestPayloadLargeBytes = 1024 * 1024;
constexpr std::size_t kNvlinkNotificationTestPayloadSmallBytes = 8;

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
#if RDMA_PROXY_HAVE_CUDA
    if (!config_.mock_mode && nvlink_notification_copy_stream_) {
        const auto set_device_status = cudaSetDevice(config_.cuda_device_id);
        if (set_device_status != cudaSuccess) {
            RDMA_PROXY_LOG_WARN(
                "cudaSetDevice failed before NVLink notification stream cleanup: ",
                cudaGetErrorString(set_device_status));
        } else {
            const auto synchronize_status = cudaStreamSynchronize(
                reinterpret_cast<cudaStream_t>(nvlink_notification_copy_stream_));
            if (synchronize_status != cudaSuccess) {
                RDMA_PROXY_LOG_WARN(
                    "cudaStreamSynchronize failed during NVLink notification stream cleanup: ",
                    cudaGetErrorString(synchronize_status));
            }
        }
    }
#endif
    destroy_cuda_stream(nvlink_notification_copy_stream_, config_.mock_mode);
    nvlink_notification_copy_stream_ = nullptr;
    for (auto& entry : buffers_) {
        free_buffer(entry.send);
        free_buffer(entry.recv);
    }
    for (auto& entry : nvlink_recv_buffers_) {
        free_buffer(entry.recv);
    }
    free_buffer(nvlink_notification_test_buffer_1mib_);
    free_buffer(nvlink_notification_test_buffer_8b_);
    free_host_staging_buffer(nvlink_notification_test_payload_1mib_);
    free_host_staging_buffer(nvlink_notification_test_payload_8b_);
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

    if (config_.nvlink_forward_expert_routing_notifications_enabled) {
        allocate_buffer(
            nvlink_notification_test_buffer_1mib_,
            kNvlinkNotificationTestPayloadLargeBytes);
        allocate_buffer(
            nvlink_notification_test_buffer_8b_,
            kNvlinkNotificationTestPayloadSmallBytes);
        allocate_host_staging_buffer(
            nvlink_notification_test_payload_1mib_,
            kNvlinkNotificationTestPayloadLargeBytes,
            uint8_t{1});
        allocate_host_staging_buffer(
            nvlink_notification_test_payload_8b_,
            kNvlinkNotificationTestPayloadSmallBytes,
            uint8_t{1});
        nvlink_notification_copy_stream_ = create_cuda_stream(
            config_.cuda_device_id,
            true,
            config_.mock_mode);
        RDMA_PROXY_LOG_INFO(
            "allocated NVLink notification test GPU buffers local_gpu=",
            config_.local_gpu_index,
            " large_bytes=", nvlink_notification_test_buffer_1mib_.bytes,
            " small_bytes=", nvlink_notification_test_buffer_8b_.bytes);
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

void CudaBuffers::copy_nvlink_notification_test_payloads_to_gpu() {
    if (!config_.nvlink_forward_expert_routing_notifications_enabled) return;
    if (!nvlink_notification_test_buffer_1mib_.ptr ||
        !nvlink_notification_test_buffer_8b_.ptr ||
        !nvlink_notification_test_payload_1mib_.ptr ||
        !nvlink_notification_test_payload_8b_.ptr ||
        nvlink_notification_test_payload_1mib_.bytes != nvlink_notification_test_buffer_1mib_.bytes ||
        nvlink_notification_test_payload_8b_.bytes != nvlink_notification_test_buffer_8b_.bytes) {
        throw std::runtime_error("NVLink notification test copy buffers are not initialized");
    }
    if (config_.mock_mode) {
        std::memcpy(
            nvlink_notification_test_buffer_1mib_.ptr,
            nvlink_notification_test_payload_1mib_.ptr,
            nvlink_notification_test_payload_1mib_.bytes);
        std::memcpy(
            nvlink_notification_test_buffer_8b_.ptr,
            nvlink_notification_test_payload_8b_.ptr,
            nvlink_notification_test_payload_8b_.bytes);
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaSetDevice(config_.cuda_device_id), "cudaSetDevice before NVLink notification test copies");
    auto stream = reinterpret_cast<cudaStream_t>(nvlink_notification_copy_stream_);
    if (!stream) throw std::runtime_error("NVLink notification copy stream is not initialized");
    check_cuda(
        cudaMemcpyAsync(
            nvlink_notification_test_buffer_1mib_.ptr,
            nvlink_notification_test_payload_1mib_.ptr,
            nvlink_notification_test_payload_1mib_.bytes,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync H2D 1 MiB NVLink notification test payload");
    check_cuda(
        cudaMemcpyAsync(
            nvlink_notification_test_buffer_8b_.ptr,
            nvlink_notification_test_payload_8b_.ptr,
            nvlink_notification_test_payload_8b_.bytes,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync H2D 8-byte NVLink notification test payload");
#else
    throw std::runtime_error("NVLink notification test copies require CUDA support or mock_mode=true");
#endif
}

std::size_t CudaBuffers::token_buffer_bytes() const {
    return config_.num_tokens * config_.token_dimension * dtype_size(config_.dtype);
}

std::size_t CudaBuffers::nvlink_receive_buffer_bytes() const {
    return token_buffer_bytes() * config_.peers.size();
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

void CudaBuffers::allocate_host_staging_buffer(
    HostStagingBuffer& buffer,
    std::size_t bytes,
    uint8_t value) {
    buffer.bytes = bytes;
    buffer.is_cuda_pinned = !config_.mock_mode;
    if (config_.mock_mode) {
        buffer.ptr = ::operator new(bytes);
    } else {
#if RDMA_PROXY_HAVE_CUDA
        check_cuda(cudaMallocHost(&buffer.ptr, bytes), "cudaMallocHost NVLink notification payload");
#else
        throw std::runtime_error("pinned NVLink notification payload allocation requires CUDA support");
#endif
    }
    std::memset(buffer.ptr, value, bytes);
}

void CudaBuffers::free_host_staging_buffer(HostStagingBuffer& buffer) {
    if (!buffer.ptr) return;
    if (buffer.is_cuda_pinned) {
#if RDMA_PROXY_HAVE_CUDA
        const auto status = cudaFreeHost(buffer.ptr);
        if (status != cudaSuccess) {
            RDMA_PROXY_LOG_WARN(
                "cudaFreeHost failed during NVLink notification payload cleanup: ",
                cudaGetErrorString(status));
        }
#endif
    } else {
        ::operator delete(buffer.ptr);
    }
    buffer.ptr = nullptr;
    buffer.bytes = 0;
    buffer.is_cuda_pinned = false;
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
