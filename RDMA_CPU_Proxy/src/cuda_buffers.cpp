#include "cuda_buffers.hpp"

#include "logging.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace rdma_proxy {
namespace {

constexpr std::size_t kRouterNotificationMapBytes = 1024;

#if RDMA_PROXY_HAVE_CUDA
void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_cuda_driver(CUresult status, const char* what) {
    if (status == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* description = nullptr;
    (void)cuGetErrorName(status, &name);
    (void)cuGetErrorString(status, &description);
    throw std::runtime_error(
        std::string(what) + ": " +
        (name ? name : "unknown CUDA driver error") +
        (description ? std::string(" (") + description + ")" : std::string{}));
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
    if (!config_.mock_mode && router_notification_publication_stream_) {
        const auto set_device_status = cudaSetDevice(config_.cuda_device_id);
        if (set_device_status != cudaSuccess) {
            RDMA_PROXY_LOG_WARN(
                "cudaSetDevice failed before router publication cleanup: ",
                cudaGetErrorString(set_device_status));
        } else {
            const auto synchronize_status = cudaStreamSynchronize(
                reinterpret_cast<cudaStream_t>(
                    router_notification_publication_stream_));
            if (synchronize_status != cudaSuccess) {
                RDMA_PROXY_LOG_WARN(
                    "cudaStreamSynchronize failed during router publication cleanup: ",
                    cudaGetErrorString(synchronize_status));
            }
        }
    }
#endif
    destroy_cuda_stream(
        router_notification_publication_stream_, config_.mock_mode);
    router_notification_publication_stream_ = nullptr;
    for (auto& entry : buffers_) {
        if (!config_.router_routing_enabled) free_buffer(entry.send);
        free_buffer(entry.recv);
    }
    free_buffer(router_send_buffer_);
    for (auto& entry : nvlink_recv_buffers_) {
        free_buffer(entry.recv);
    }
    free_buffer(router_notification_publication_buffers_.device_map);
    free_buffer(router_notification_publication_buffers_.device_flag);
    free_pinned_buffer(router_notification_publication_buffers_.host_map);
    free_pinned_buffer(router_notification_publication_buffers_.host_flag);
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
    if (config_.router_routing_enabled) {
        allocate_buffer(router_send_buffer_, bytes);
        RDMA_PROXY_LOG_INFO("allocated shared router RDMA send buffer bytes=", bytes);
    }
    for (const auto& peer : config_.peers) {
        PeerGpuBuffers entry;
        entry.peer_rank = peer.node_rank;
        if (config_.router_routing_enabled) {
            entry.send = router_send_buffer_;
        } else {
            allocate_buffer(entry.send, bytes);
        }
        allocate_buffer(entry.recv, bytes);
        buffers_.push_back(entry);
        RDMA_PROXY_LOG_INFO("allocated GPU buffers for peer ", peer.node_rank, " bytes=", bytes);
    }

    nvlink_recv_buffers_.clear();
    if (config_.nvlink_forwarding_enabled) {
        const auto forwarding_bytes = nvlink_receive_buffer_bytes();
        if (config_.router_routing_enabled) {
            const auto global_gpu_count = static_cast<std::size_t>(config_.num_nodes) *
                static_cast<std::size_t>(config_.num_gpus_per_node);
            nvlink_recv_buffers_.reserve(global_gpu_count);
            for (int source_node = 0; source_node < config_.num_nodes; ++source_node) {
                for (int source_gpu = 0; source_gpu < config_.num_gpus_per_node; ++source_gpu) {
                    NvlinkReceiveBuffer entry;
                    entry.source_node_rank = source_node;
                    entry.source_gpu_index = source_gpu;
                    allocate_buffer(entry.recv, forwarding_bytes);
                    nvlink_recv_buffers_.push_back(entry);
                    RDMA_PROXY_LOG_INFO(
                        "allocated router NVLink receive buffer local_gpu=", config_.local_gpu_index,
                        " source_node=", source_node,
                        " source_gpu=", source_gpu,
                        " bytes=", forwarding_bytes);
                }
            }
        } else {
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
    }

    if (config_.router_routing_enabled &&
        config_.nvlink_forward_completion_notifications_enabled) {
        auto& publication = router_notification_publication_buffers_;
        allocate_pinned_buffer(publication.host_map, kRouterNotificationMapBytes);
        allocate_pinned_buffer(publication.host_flag, sizeof(uint32_t));
        allocate_buffer(publication.device_map, kRouterNotificationMapBytes);
        allocate_buffer(publication.device_flag, sizeof(uint32_t));
        router_notification_publication_stream_ = create_cuda_stream(
            config_.cuda_device_id,
            /*nonblocking=*/true,
            config_.mock_mode);
        RDMA_PROXY_LOG_INFO(
            "allocated router notification publication buffers map_bytes=",
            kRouterNotificationMapBytes,
            " flag_bytes=", sizeof(uint32_t));
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

void CudaBuffers::fill_router_test_pattern(int source_rank, uint64_t iteration) {
    if (!config_.router_routing_enabled || !router_send_buffer_.ptr) {
        throw std::runtime_error("router send buffer is not initialized");
    }
    const auto pattern = make_test_pattern(
        router_send_buffer_.bytes, source_rank, -1, config_.local_gpu_index, iteration);
    if (config_.mock_mode) {
        std::memcpy(router_send_buffer_.ptr, pattern.data(), pattern.size());
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaMemcpy(
        router_send_buffer_.ptr, pattern.data(), pattern.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D router test pattern");
#else
    throw std::runtime_error("CUDA router test-pattern fill requested but CUDA support was not built");
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

const NvlinkReceiveBuffer& CudaBuffers::nvlink_receive_buffer_for_source(
    int source_node_rank,
    int source_gpu_index) const {
    auto it = std::find_if(nvlink_recv_buffers_.begin(), nvlink_recv_buffers_.end(), [&](const auto& entry) {
        return entry.source_node_rank == source_node_rank &&
            entry.source_gpu_index == source_gpu_index;
    });
    if (it == nvlink_recv_buffers_.end()) {
        throw std::runtime_error("unknown NVLink source node/GPU");
    }
    return *it;
}

void CudaBuffers::install_expert_metadata(RouterExpertMetadata metadata) {
    if (!config_.router_routing_enabled || !config_.nvlink_forwarding_enabled) {
        throw std::runtime_error(
            "expert metadata requires combined router/NVLink mode");
    }
    if (metadata.destination_node_rank != config_.node_rank ||
        metadata.destination_gpu_index != config_.local_gpu_index) {
        throw std::runtime_error("expert metadata destination does not match this proxy");
    }
    const int expected_experts_per_gpu = config_.router_num_experts /
        (config_.num_nodes * config_.num_gpus_per_node);
    const int expected_first_expert =
        (config_.node_rank * config_.num_gpus_per_node + config_.local_gpu_index) *
        expected_experts_per_gpu;
    if (metadata.num_nodes != config_.num_nodes ||
        metadata.num_gpus_per_node != config_.num_gpus_per_node ||
        metadata.num_experts != config_.router_num_experts ||
        metadata.experts_per_gpu != expected_experts_per_gpu ||
        metadata.first_global_expert != expected_first_expert ||
        metadata.num_tokens != config_.num_tokens) {
        throw std::runtime_error("expert metadata dimensions do not match this proxy");
    }
    if (metadata.expert_offsets.size() !=
            static_cast<std::size_t>(expected_experts_per_gpu + 1) ||
        metadata.expert_offsets.front() != 0 ||
        metadata.expert_offsets.back() < 0 ||
        static_cast<std::size_t>(metadata.expert_offsets.back()) !=
            metadata.expert_token_indices.size() ||
        !std::is_sorted(metadata.expert_offsets.begin(), metadata.expert_offsets.end())) {
        throw std::runtime_error("expert metadata offsets are invalid");
    }
    if (std::any_of(
            metadata.expert_token_indices.begin(),
            metadata.expert_token_indices.end(),
            [&](int32_t index) {
                return index < 0 ||
                    static_cast<std::size_t>(index) >= config_.num_tokens;
            })) {
        throw std::runtime_error("expert metadata contains an invalid token index");
    }
    for (int expert = 0; expert < expected_experts_per_gpu; ++expert) {
        const auto begin = metadata.expert_token_indices.begin() +
            metadata.expert_offsets[static_cast<std::size_t>(expert)];
        const auto end = metadata.expert_token_indices.begin() +
            metadata.expert_offsets[static_cast<std::size_t>(expert + 1)];
        if (!std::is_sorted(begin, end)) {
            throw std::runtime_error(
                "expert metadata token indices are not monotonic within an expert list");
        }
    }
    if (metadata.source_node_rank < 0 ||
        metadata.source_node_rank >= config_.num_nodes ||
        metadata.source_gpu_index < 0 ||
        metadata.source_gpu_index >= config_.num_gpus_per_node) {
        throw std::runtime_error("expert metadata source is out of range");
    }
    std::lock_guard<std::mutex> lock(expert_token_heads_mutex_);
    auto it = std::find_if(
        nvlink_recv_buffers_.begin(), nvlink_recv_buffers_.end(),
        [&](const auto& entry) {
            return entry.source_node_rank == metadata.source_node_rank &&
                entry.source_gpu_index == metadata.source_gpu_index;
        });
    if (it == nvlink_recv_buffers_.end()) {
        throw std::runtime_error("missing source-specific NVLink receive buffer");
    }
    if (it->expert_metadata_ready) {
        throw std::runtime_error("duplicate expert metadata for source GPU");
    }
    it->expert_metadata = std::move(metadata);
    it->expert_metadata_ready = true;
    it->expert_token_head_state = RouterExpertTokenHeadState{};
    it->expert_token_head_state.expert_token_heads.assign(
        static_cast<std::size_t>(expected_experts_per_gpu), 0);
}

RouterExpertTokenHeadState CudaBuffers::expert_token_head_state_for_source(
    int source_node_rank,
    int source_gpu_index) const {
    std::lock_guard<std::mutex> lock(expert_token_heads_mutex_);
    const auto it = std::find_if(
        nvlink_recv_buffers_.begin(), nvlink_recv_buffers_.end(),
        [&](const auto& entry) {
            return entry.source_node_rank == source_node_rank &&
                entry.source_gpu_index == source_gpu_index;
        });
    if (it == nvlink_recv_buffers_.end()) {
        throw std::runtime_error("unknown NVLink source node/GPU");
    }
    if (!it->expert_metadata_ready) {
        throw std::runtime_error("router expert metadata is not ready for source GPU");
    }
    return it->expert_token_head_state;
}

void CudaBuffers::update_expert_token_heads(
    int source_node_rank,
    int source_gpu_index,
    uint64_t iteration,
    std::size_t start_token,
    std::size_t num_tokens) {
    if (!config_.router_routing_enabled || !config_.nvlink_forwarding_enabled) {
        throw std::runtime_error(
            "expert token heads require combined router/NVLink mode");
    }
    if (start_token > config_.num_tokens ||
        num_tokens > config_.num_tokens - start_token) {
        throw std::runtime_error("received NVLink token range exceeds buffer capacity");
    }

    std::lock_guard<std::mutex> lock(expert_token_heads_mutex_);
    auto it = std::find_if(
        nvlink_recv_buffers_.begin(), nvlink_recv_buffers_.end(),
        [&](const auto& entry) {
            return entry.source_node_rank == source_node_rank &&
                entry.source_gpu_index == source_gpu_index;
        });
    if (it == nvlink_recv_buffers_.end()) {
        throw std::runtime_error("unknown NVLink source node/GPU");
    }
    if (!it->expert_metadata_ready) {
        throw std::runtime_error("router expert metadata is not ready for source GPU");
    }

    auto& state = it->expert_token_head_state;
    const auto& metadata = it->expert_metadata;
    if (!state.iteration_initialized || iteration > state.iteration) {
        if (start_token != 0) {
            throw std::runtime_error(
                "first NVLink notification for an iteration does not start at token zero");
        }
        state.iteration_initialized = true;
        state.iteration = iteration;
        state.received_token_frontier = 0;
        std::fill(
            state.expert_token_heads.begin(),
            state.expert_token_heads.end(),
            std::size_t{0});
    } else if (iteration < state.iteration) {
        throw std::runtime_error("received stale NVLink forwarding notification");
    }

    if (start_token != state.received_token_frontier) {
        throw std::runtime_error(
            "NVLink forwarding notifications are not contiguous for source buffer");
    }
    const auto received_end = start_token + num_tokens;
    if (state.expert_token_heads.size() !=
        static_cast<std::size_t>(metadata.experts_per_gpu)) {
        throw std::runtime_error("expert token head count does not match metadata");
    }

    for (int expert = 0; expert < metadata.experts_per_gpu; ++expert) {
        const auto expert_index = static_cast<std::size_t>(expert);
        const auto list_begin = static_cast<std::size_t>(
            metadata.expert_offsets[expert_index]);
        const auto list_end = static_cast<std::size_t>(
            metadata.expert_offsets[expert_index + 1]);
        auto& head = state.expert_token_heads[expert_index];
        while (list_begin + head < list_end &&
               static_cast<std::size_t>(
                   metadata.expert_token_indices[list_begin + head]) < received_end) {
            ++head;
        }
    }
    state.received_token_frontier = received_end;
}

void CudaBuffers::process_router_notification_completion(
    int source_node_rank,
    int source_gpu_index,
    uint64_t iteration,
    std::size_t start_token,
    std::size_t num_tokens) {
    if (!config_.nvlink_forward_notification_flush_only_enabled) {
        update_expert_token_heads(
            source_node_rank,
            source_gpu_index,
            iteration,
            start_token,
            num_tokens);
    }
    flush_router_notification_publication();
}

void CudaBuffers::flush_router_notification_publication() {
    auto& publication = router_notification_publication_buffers_;
    if (!publication.host_map.ptr || !publication.host_flag.ptr ||
        !publication.device_map.ptr || !publication.device_flag.ptr ||
        publication.host_map.bytes != kRouterNotificationMapBytes ||
        publication.device_map.bytes != kRouterNotificationMapBytes ||
        publication.host_flag.bytes != sizeof(uint32_t) ||
        publication.device_flag.bytes != sizeof(uint32_t)) {
        throw std::runtime_error(
            "router notification publication buffers are not initialized");
    }

    if (config_.mock_mode) {
        std::memcpy(
            publication.device_map.ptr,
            publication.host_map.ptr,
            publication.host_map.bytes);
        std::memcpy(
            publication.device_flag.ptr,
            publication.host_flag.ptr,
            publication.host_flag.bytes);
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    if (!router_notification_publication_stream_) {
        throw std::runtime_error(
            "router notification publication stream is not initialized");
    }
    auto stream = reinterpret_cast<cudaStream_t>(
        router_notification_publication_stream_);
    check_cuda(cudaMemcpyAsync(
        publication.device_map.ptr,
        publication.host_map.ptr,
        publication.host_map.bytes,
        cudaMemcpyHostToDevice,
        stream), "cudaMemcpyAsync H2D router notification map");

    // A stream memory write is host-asynchronous, writes the aligned flag as
    // one 32-bit value, and (without NO_MEMORY_BARRIER) performs a system-wide
    // fence after the preceding map copy and before publishing the new flag.
    const auto flag_value = *static_cast<const uint32_t*>(
        publication.host_flag.ptr);
    check_cuda_driver(cuStreamWriteValue32(
        reinterpret_cast<CUstream>(stream),
        reinterpret_cast<CUdeviceptr>(publication.device_flag.ptr),
        flag_value,
        CU_STREAM_WRITE_VALUE_DEFAULT),
        "cuStreamWriteValue32 router notification flag");
#else
    throw std::runtime_error(
        "CUDA router notification publication requested but CUDA support was not built");
#endif
}

std::size_t CudaBuffers::token_buffer_bytes() const {
    return config_.num_tokens * config_.token_dimension * dtype_size(config_.dtype);
}

std::size_t CudaBuffers::nvlink_receive_buffer_bytes() const {
    if (config_.router_routing_enabled) {
        return token_buffer_bytes();
    }
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

void CudaBuffers::allocate_pinned_buffer(
    CpuPinnedBuffer& buffer,
    std::size_t bytes) {
    buffer.bytes = bytes;
    buffer.is_mock_host_memory = config_.mock_mode;
    if (config_.mock_mode) {
        buffer.ptr = ::operator new(bytes);
        std::memset(buffer.ptr, 0, bytes);
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaMallocHost(&buffer.ptr, bytes), "cudaMallocHost");
    std::memset(buffer.ptr, 0, bytes);
#else
    throw std::runtime_error(
        "CUDA pinned allocation requested but CUDA support was not built");
#endif
}

void CudaBuffers::free_pinned_buffer(CpuPinnedBuffer& buffer) {
    if (!buffer.ptr) return;
    if (buffer.is_mock_host_memory) {
        ::operator delete(buffer.ptr);
    } else {
#if RDMA_PROXY_HAVE_CUDA
        const auto status = cudaFreeHost(buffer.ptr);
        if (status != cudaSuccess) {
            RDMA_PROXY_LOG_WARN(
                "cudaFreeHost failed during cleanup: ",
                cudaGetErrorString(status));
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
