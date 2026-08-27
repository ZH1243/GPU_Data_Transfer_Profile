#include "cuda_buffers.hpp"

#include "logging.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda.h>
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

DeviceBufferElementType device_buffer_element_type(DataType dtype) {
    switch (dtype) {
        case DataType::kBF16: return DeviceBufferElementType::kBF16;
        case DataType::kFP16: return DeviceBufferElementType::kFP16;
        case DataType::kFP32: return DeviceBufferElementType::kFP32;
    }
    throw std::runtime_error("unsupported device-buffer data type");
}

DeviceBufferAllocationRequest token_buffer_request(
    DeviceBufferKind kind,
    const ProxyConfig& config,
    std::size_t bytes,
    int peer_rank = -1,
    int source_node_rank = -1,
    int source_gpu_index = -1) {
    DeviceBufferAllocationRequest request;
    request.kind = kind;
    request.element_type = device_buffer_element_type(config.dtype);
    request.peer_rank = peer_rank;
    request.source_node_rank = source_node_rank;
    request.source_gpu_index = source_gpu_index;
    request.bytes = bytes;
    const auto row_bytes = config.token_dimension * dtype_size(config.dtype);
    if (row_bytes == 0 || bytes % row_bytes != 0) {
        throw std::runtime_error("device token-buffer size is not row aligned");
    }
    request.dimensions = {bytes / row_bytes, config.token_dimension};
    return request;
}

}  // namespace

#if RDMA_PROXY_HAVE_CUDA
void launch_copy_tokens_kernel(void* dst, const void* src, std::size_t bytes);
#endif

CudaBuffers::CudaBuffers(ProxyConfig config) : config_(std::move(config)) {}

void CudaBuffers::set_external_device_buffer_allocator(
    ExternalDeviceBufferAllocator allocator) {
    if (!buffers_.empty() || router_send_buffer_.ptr ||
        !nvlink_recv_buffers_.empty() ||
        router_notification_publication_buffers_.device_ready_rows.ptr) {
        throw std::runtime_error(
            "external device-buffer allocator must be set before initialization");
    }
    external_device_buffer_allocator_ = std::move(allocator);
}

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
    if (!config_.mock_mode) {
#if RDMA_PROXY_HAVE_CUDA
        if (router_computation_start_event_) {
            const auto status = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(
                router_computation_start_event_));
            if (status != cudaSuccess) {
                RDMA_PROXY_LOG_WARN(
                    "cudaEventDestroy(start) failed during cleanup: ",
                    cudaGetErrorString(status));
            }
        }
        if (router_computation_end_event_) {
            const auto status = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(
                router_computation_end_event_));
            if (status != cudaSuccess) {
                RDMA_PROXY_LOG_WARN(
                    "cudaEventDestroy(end) failed during cleanup: ",
                    cudaGetErrorString(status));
            }
        }
#endif
    }
    router_computation_start_event_ = nullptr;
    router_computation_end_event_ = nullptr;
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
        free_buffer(entry.expert_token_indices_device);
    }
    free_buffer(router_notification_publication_buffers_.device_table);
    free_buffer(router_notification_publication_buffers_.device_ready_rows);
    free_pinned_buffer(router_notification_publication_buffers_.host_table);
    free_pinned_buffer(router_notification_publication_buffers_.host_ready_rows);
    free_pinned_buffer(router_notification_ready_values_);
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
        allocate_buffer(
            router_send_buffer_,
            token_buffer_request(DeviceBufferKind::kRdmaSend, config_, bytes));
        RDMA_PROXY_LOG_INFO("allocated shared router RDMA send buffer bytes=", bytes);
    }
    for (const auto& peer : config_.peers) {
        PeerGpuBuffers entry;
        entry.peer_rank = peer.node_rank;
        if (config_.router_routing_enabled) {
            entry.send = router_send_buffer_;
        } else {
            allocate_buffer(
                entry.send,
                token_buffer_request(
                    DeviceBufferKind::kRdmaSend, config_, bytes, peer.node_rank));
        }
        allocate_buffer(
            entry.recv,
            token_buffer_request(
                DeviceBufferKind::kRdmaReceive, config_, bytes, peer.node_rank));
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
                    const auto request = token_buffer_request(
                        DeviceBufferKind::kNvlinkReceive,
                        config_,
                        forwarding_bytes,
                        -1,
                        source_node,
                        source_gpu);
                    if (source_node != config_.node_rank &&
                        source_gpu == config_.local_gpu_index) {
                        alias_buffer(
                            entry.recv,
                            buffers_for_peer(source_node).recv,
                            request);
                        RDMA_PROXY_LOG_INFO(
                            "aliased router direct-RDMA input local_gpu=",
                            config_.local_gpu_index,
                            " source_node=", source_node,
                            " source_gpu=", source_gpu,
                            " bytes=", forwarding_bytes);
                    } else {
                        allocate_buffer(entry.recv, request);
                    }
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
                allocate_buffer(
                    entry.recv,
                    token_buffer_request(
                        DeviceBufferKind::kNvlinkReceive,
                        config_,
                        forwarding_bytes,
                        -1,
                        -1,
                        source_gpu));
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
        allocate_pinned_buffer(publication.host_ready_rows, sizeof(int32_t));
        allocate_buffer(
            publication.device_ready_rows,
            DeviceBufferAllocationRequest{
                DeviceBufferKind::kGatherReadyRows,
                DeviceBufferElementType::kInt32,
                -1,
                -1,
                -1,
                sizeof(int32_t),
                {1}});
        router_notification_publication_stream_ = create_cuda_stream(
            config_.cuda_device_id,
            /*nonblocking=*/true,
            config_.mock_mode);
#if RDMA_PROXY_HAVE_CUDA
        if (!config_.mock_mode) {
            cudaEvent_t start_event = nullptr;
            cudaEvent_t end_event = nullptr;
            check_cuda(cudaEventCreate(&start_event),
                       "cudaEventCreate QuACK computation start");
            router_computation_start_event_ = start_event;
            check_cuda(cudaEventCreate(&end_event),
                       "cudaEventCreate QuACK computation end");
            router_computation_end_event_ = end_event;
        }
#endif
        RDMA_PROXY_LOG_INFO(
            "allocated QuACK gather ready-row flag bytes=", sizeof(int32_t),
            "; table and A_idx allocation waits for all expert metadata");
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
    it->expert_token_head_state.expert_token_tails.assign(
        static_cast<std::size_t>(expected_experts_per_gpu), 0);

    if (config_.nvlink_forward_completion_notifications_enabled &&
        std::all_of(
            nvlink_recv_buffers_.begin(), nvlink_recv_buffers_.end(),
            [](const NvlinkReceiveBuffer& entry) {
                return entry.expert_metadata_ready;
            })) {
        initialize_router_computation_scheduler_locked();
    }
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

RouterComputationSchedulerState
CudaBuffers::router_computation_scheduler_state() const {
    std::lock_guard<std::mutex> lock(expert_token_heads_mutex_);
    return router_computation_scheduler_state_;
}

bool CudaBuffers::is_router_computation_input(
    const NvlinkReceiveBuffer& buffer) const {
    if (!config_.router_computation_forwarded_inputs_only) return true;
    return buffer.source_node_rank >= 0 &&
        buffer.source_node_rank != config_.node_rank &&
        buffer.source_gpu_index >= 0;
}

void CudaBuffers::initialize_router_computation_scheduler_locked() {
    if (router_computation_scheduler_state_.initialized) {
        throw std::runtime_error("router computation scheduler is already initialized");
    }
    if (config_.expert_gemm_m_tile == 0 || config_.expert_gemm_n_tile == 0 ||
        config_.expert_gemm_dimension == 0 || config_.expert_gemm_cluster_m == 0 ||
        config_.expert_gemm_max_swizzle_size == 0) {
        throw std::runtime_error("invalid expert GEMM tile configuration");
    }
    if (config_.expert_gemm_m_tile >
        std::numeric_limits<std::size_t>::max() / config_.expert_gemm_cluster_m) {
        throw std::runtime_error("expert GEMM M-cluster size overflows size_t");
    }
    const auto m_cluster_tokens =
        config_.expert_gemm_m_tile * config_.expert_gemm_cluster_m;
    const auto clusters_n = config_.expert_gemm_dimension / config_.expert_gemm_n_tile +
        (config_.expert_gemm_dimension % config_.expert_gemm_n_tile != 0 ? 1 : 0);
    if (clusters_n >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("QuACK gather-table cid_n_base exceeds int32 range");
    }
    const auto group_size =
        std::min(config_.expert_gemm_max_swizzle_size, clusters_n);
    if (group_size == 0 || clusters_n % group_size != 0) {
        throw std::runtime_error(
            "expert GEMM N clusters are not divisible by the QuACK swizzle group size");
    }
    const auto n_groups_per_m_cluster = clusters_n / group_size;

    const auto experts_per_gpu = static_cast<std::size_t>(
        config_.router_num_experts /
        (config_.num_nodes * config_.num_gpus_per_node));
    auto& state = router_computation_scheduler_state_;
    state = RouterComputationSchedulerState{};
    state.expert_total_tokens.assign(experts_per_gpu, 0);
    state.expert_num_ready_tokens.assign(experts_per_gpu, 0);
    state.expert_num_notified_batches.assign(experts_per_gpu, 0);
    state.expert_total_batches.assign(experts_per_gpu, 0);

    router_computation_buffer_indices_.clear();
    for (std::size_t index = 0; index < nvlink_recv_buffers_.size(); ++index) {
        if (is_router_computation_input(nvlink_recv_buffers_[index])) {
            router_computation_buffer_indices_.push_back(index);
        }
    }
    if (router_computation_buffer_indices_.empty()) {
        throw std::runtime_error("QuACK gather scheduler has no active input buffers");
    }

    std::size_t a_idx_capacity = 0;
    for (const auto buffer_index : router_computation_buffer_indices_) {
        const auto& buffer = nvlink_recv_buffers_[buffer_index];
        if (!buffer.expert_metadata_ready) {
            throw std::runtime_error(
                "cannot initialize router computation scheduler before metadata exchange completes");
        }
        const auto& offsets = buffer.expert_metadata.expert_offsets;
        for (std::size_t expert = 0; expert < experts_per_gpu; ++expert) {
            const auto count = static_cast<std::size_t>(
                offsets[expert + 1] - offsets[expert]);
            auto& total = state.expert_total_tokens[expert];
            if (count > std::numeric_limits<std::size_t>::max() - total) {
                throw std::runtime_error("local expert token total overflows size_t");
            }
            total += count;
        }
        a_idx_capacity = std::max(
            a_idx_capacity, buffer.expert_metadata.expert_token_indices.size());
    }

    // QuACK requires every A_idx_j tensor to have the same shape. Keep each
    // meaningful expert-major index list in the prefix and zero-pad the rest.
    // A one-element allocation also gives empty-input configurations a stable
    // pointer/shape, although no gather-table row references that padding.
    a_idx_capacity = std::max<std::size_t>(a_idx_capacity, 1);
    if (a_idx_capacity > std::numeric_limits<std::size_t>::max() / sizeof(int32_t)) {
        throw std::runtime_error("QuACK A_idx allocation size overflows size_t");
    }
    for (const auto buffer_index : router_computation_buffer_indices_) {
        auto& buffer = nvlink_recv_buffers_[buffer_index];
        buffer.expert_token_index_count =
            buffer.expert_metadata.expert_token_indices.size();
        allocate_buffer(
            buffer.expert_token_indices_device,
            DeviceBufferAllocationRequest{
                DeviceBufferKind::kRouterAIdx,
                DeviceBufferElementType::kInt32,
                -1,
                buffer.source_node_rank,
                buffer.source_gpu_index,
                a_idx_capacity * sizeof(int32_t),
                {a_idx_capacity}});
        std::vector<int32_t> padded_indices(a_idx_capacity, 0);
        std::copy(
            buffer.expert_metadata.expert_token_indices.begin(),
            buffer.expert_metadata.expert_token_indices.end(),
            padded_indices.begin());
        if (config_.mock_mode) {
            std::memcpy(
                buffer.expert_token_indices_device.ptr,
                padded_indices.data(),
                padded_indices.size() * sizeof(int32_t));
        } else {
#if RDMA_PROXY_HAVE_CUDA
            check_cuda(cudaMemcpy(
                buffer.expert_token_indices_device.ptr,
                padded_indices.data(),
                padded_indices.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice), "cudaMemcpy H2D QuACK A_idx");
#else
            throw std::runtime_error(
                "QuACK A_idx initialization requested but CUDA support was not built");
#endif
        }
    }

    std::size_t total_batches = 0;
    for (std::size_t expert = 0; expert < experts_per_gpu; ++expert) {
        const auto tokens = state.expert_total_tokens[expert];
        const auto batches = tokens / m_cluster_tokens +
            (tokens % m_cluster_tokens != 0 ? 1 : 0);
        state.expert_total_batches[expert] = batches;
        if (batches > std::numeric_limits<std::size_t>::max() - total_batches) {
            throw std::runtime_error("router computation batch count overflows size_t");
        }
        total_batches += batches;
    }

    const auto receive_buffer_count = router_computation_buffer_indices_.size();
    auto& publication = router_notification_publication_buffers_;
    if (receive_buffer_count >
        (std::numeric_limits<std::size_t>::max() - 2) / 2) {
        throw std::runtime_error("QuACK gather-table width overflows size_t");
    }
    publication.table_width = 2 + 2 * receive_buffer_count;
    if (publication.table_width >
        std::numeric_limits<std::size_t>::max() / sizeof(int32_t)) {
        throw std::runtime_error("QuACK gather-table row size overflows size_t");
    }
    publication.table_row_bytes = publication.table_width * sizeof(int32_t);
    if (total_batches != 0 &&
        n_groups_per_m_cluster >
            std::numeric_limits<std::size_t>::max() / total_batches) {
        throw std::runtime_error("QuACK gather-table row count overflows size_t");
    }
    publication.table_rows = total_batches * n_groups_per_m_cluster;
    // Flush-only mode is explicitly defined to publish the first row even if
    // this GPU happens to receive no expert tokens.
    if (publication.table_rows == 0 &&
        config_.nvlink_forward_notification_flush_only_enabled) {
        publication.table_rows = 1;
    }
    publication.num_input_buffers = receive_buffer_count;
    publication.a_idx_capacity = a_idx_capacity;
    publication.n_groups_per_m_cluster = n_groups_per_m_cluster;
    publication.group_size = static_cast<int32_t>(group_size);
    if (publication.table_rows >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("QuACK ready-row prefix exceeds int32 range");
    }
    if (publication.table_rows != 0 &&
        publication.table_row_bytes >
            std::numeric_limits<std::size_t>::max() /
                publication.table_rows) {
        throw std::runtime_error("QuACK gather-table allocation size overflows size_t");
    }
    const auto table_bytes = publication.table_rows * publication.table_row_bytes;
    if (table_bytes != 0) {
        allocate_pinned_buffer(publication.host_table, table_bytes);
        allocate_buffer(
            publication.device_table,
            DeviceBufferAllocationRequest{
                DeviceBufferKind::kGatherTable,
                DeviceBufferElementType::kInt32,
                -1,
                -1,
                -1,
                table_bytes,
                {publication.table_rows, publication.table_width}});
    }
    if (config_.nvlink_forward_notification_flag_update_mode ==
        NvlinkForwardNotificationFlagUpdateMode::kMemcpy) {
        const auto ready_value_count = publication.table_rows + 1;
        allocate_pinned_buffer(
            router_notification_ready_values_,
            ready_value_count * sizeof(int32_t));
        auto* ready_values = static_cast<int32_t*>(
            router_notification_ready_values_.ptr);
        for (std::size_t ready_rows = 0;
             ready_rows < ready_value_count;
             ++ready_rows) {
            ready_values[ready_rows] = static_cast<int32_t>(ready_rows);
        }
    }
    state.initialized = true;
    RDMA_PROXY_LOG_INFO(
        "initialized QuACK gather-table scheduler local_experts=",
        experts_per_gpu,
        " receive_buffers=", receive_buffer_count,
        " m_clusters=", total_batches,
        " n_groups_per_m_cluster=", n_groups_per_m_cluster,
        " group_size=", publication.group_size,
        " table_rows=", publication.table_rows,
        " table_width=", publication.table_width,
        " table_bytes=", table_bytes,
        " a_idx_capacity=", a_idx_capacity);
}

void CudaBuffers::reset_router_computation_iteration_locked(uint64_t iteration) {
    auto& scheduler = router_computation_scheduler_state_;
    if (!scheduler.initialized) {
        throw std::runtime_error("router computation scheduler is not initialized");
    }
    const bool reset_existing_iteration = scheduler.iteration_initialized;
    if (reset_existing_iteration) {
        // Rows are immutable within an iteration, but a new iteration reuses
        // them. Wait for prior H2D reads before clearing pinned host memory.
        synchronize_cuda_stream(
            router_notification_publication_stream_, config_.mock_mode);
    }
    for (auto& buffer : nvlink_recv_buffers_) {
        auto& progress = buffer.expert_token_head_state;
        progress.iteration_initialized = false;
        progress.iteration = iteration;
        progress.received_token_frontier = 0;
        std::fill(progress.expert_token_heads.begin(),
                  progress.expert_token_heads.end(), std::size_t{0});
        std::fill(progress.expert_token_tails.begin(),
                  progress.expert_token_tails.end(), std::size_t{0});
    }
    std::fill(scheduler.expert_num_ready_tokens.begin(),
              scheduler.expert_num_ready_tokens.end(), std::size_t{0});
    std::fill(scheduler.expert_num_notified_batches.begin(),
              scheduler.expert_num_notified_batches.end(), std::size_t{0});
    scheduler.iteration_initialized = true;
    scheduler.iteration = iteration;
    scheduler.published_batches = 0;
    scheduler.published_rows = 0;

    {
        std::lock_guard<std::mutex> timing_lock(
            router_computation_timing_mutex_);
        router_computation_timing_initialized_ = true;
        router_computation_timing_iteration_ = iteration;
        router_computation_start_recorded_ = false;
        router_computation_end_recorded_ = false;
    }

    auto& publication = router_notification_publication_buffers_;
    if (publication.host_table.ptr && publication.host_table.bytes != 0) {
        std::memset(publication.host_table.ptr, 0, publication.host_table.bytes);
    }
    *static_cast<int32_t*>(publication.host_ready_rows.ptr) = 0;
    // Publish zero before any row for this iteration. A QuACK persistent
    // scheduler must never observe the previous iteration's ready prefix.
    flush_router_notification_publication_range(0, 0);
}

void CudaBuffers::begin_router_notification_iteration(uint64_t iteration) {
    if (!config_.router_routing_enabled ||
        !config_.nvlink_forward_completion_notifications_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(expert_token_heads_mutex_);
    auto& scheduler = router_computation_scheduler_state_;
    if (!scheduler.initialized) {
        throw std::runtime_error("router computation scheduler is not initialized");
    }
    if (scheduler.iteration_initialized && iteration < scheduler.iteration) {
        throw std::runtime_error("cannot begin a stale router notification iteration");
    }
    if (!scheduler.iteration_initialized || iteration > scheduler.iteration) {
        reset_router_computation_iteration_locked(iteration);
    }
}

void CudaBuffers::synchronize_router_notification_publication() {
    if (!config_.router_routing_enabled ||
        !config_.nvlink_forward_completion_notifications_enabled) {
        return;
    }
    synchronize_cuda_stream(
        router_notification_publication_stream_, config_.mock_mode);
}

void CudaBuffers::record_router_computation_end(
    uint64_t iteration,
    uintptr_t cuda_stream) {
    std::lock_guard<std::mutex> lock(router_computation_timing_mutex_);
    if (!router_computation_timing_initialized_ ||
        router_computation_timing_iteration_ != iteration) {
        throw std::runtime_error(
            "router computation timing iteration is not prepared");
    }
    if (router_computation_end_recorded_) {
        throw std::runtime_error(
            "router computation end event is already recorded");
    }
    if (config_.mock_mode) {
        router_computation_end_recorded_ = true;
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    if (!router_computation_end_event_ || cuda_stream == 0) {
        throw std::runtime_error(
            "router computation end event or CUDA stream is invalid");
    }
    check_cuda(
        cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(router_computation_end_event_),
            reinterpret_cast<cudaStream_t>(cuda_stream)),
        "cudaEventRecord QuACK computation end");
    router_computation_end_recorded_ = true;
#else
    (void)cuda_stream;
    throw std::runtime_error(
        "router computation timing requested without CUDA support");
#endif
}

float CudaBuffers::router_computation_elapsed_ms(uint64_t iteration) {
    std::lock_guard<std::mutex> lock(router_computation_timing_mutex_);
    if (!router_computation_timing_initialized_ ||
        router_computation_timing_iteration_ != iteration ||
        !router_computation_start_recorded_ ||
        !router_computation_end_recorded_) {
        throw std::runtime_error(
            "router computation timing events are incomplete");
    }
    if (config_.mock_mode) {
        throw std::runtime_error(
            "GPU computation elapsed time is unavailable in mock mode");
    }
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(
        cudaEventSynchronize(
            reinterpret_cast<cudaEvent_t>(router_computation_end_event_)),
        "cudaEventSynchronize QuACK computation end");
    float elapsed_ms = 0.0F;
    check_cuda(
        cudaEventElapsedTime(
            &elapsed_ms,
            reinterpret_cast<cudaEvent_t>(router_computation_start_event_),
            reinterpret_cast<cudaEvent_t>(router_computation_end_event_)),
        "cudaEventElapsedTime QuACK computation");
    return elapsed_ms;
#else
    throw std::runtime_error(
        "router computation timing requested without CUDA support");
#endif
}

std::size_t CudaBuffers::router_computation_num_tokens() const {
    std::lock_guard<std::mutex> lock(expert_token_heads_mutex_);
    if (!router_computation_scheduler_state_.initialized) {
        throw std::runtime_error(
            "router computation scheduler is not initialized");
    }
    std::size_t total = 0;
    for (const auto count :
         router_computation_scheduler_state_.expert_total_tokens) {
        if (count > std::numeric_limits<std::size_t>::max() - total) {
            throw std::runtime_error("router computation token total overflows size_t");
        }
        total += count;
    }
    return total;
}

void CudaBuffers::update_expert_token_heads_locked(
    NvlinkReceiveBuffer& buffer,
    uint64_t iteration,
    std::size_t start_token,
    std::size_t num_tokens,
    bool flush_ready_entries_per_entry) {
    auto& scheduler = router_computation_scheduler_state_;
    if (!scheduler.iteration_initialized || iteration > scheduler.iteration) {
        if (start_token != 0) {
            throw std::runtime_error(
                "first NVLink notification for an iteration does not start at token zero");
        }
        reset_router_computation_iteration_locked(iteration);
    } else if (iteration < scheduler.iteration) {
        throw std::runtime_error("received stale NVLink forwarding notification");
    }

    auto& progress = buffer.expert_token_head_state;
    const auto& metadata = buffer.expert_metadata;
    if (!progress.iteration_initialized) {
        if (start_token != 0) {
            throw std::runtime_error(
                "first NVLink notification for a source buffer does not start at token zero");
        }
        progress.iteration_initialized = true;
        progress.iteration = iteration;
    } else if (progress.iteration != iteration) {
        throw std::runtime_error("NVLink source-buffer iteration state mismatch");
    }
    if (start_token != progress.received_token_frontier) {
        throw std::runtime_error(
            "NVLink forwarding notifications are not contiguous for source buffer");
    }

    const auto received_end = start_token + num_tokens;
    if (progress.expert_token_heads.size() !=
            static_cast<std::size_t>(metadata.experts_per_gpu) ||
        progress.expert_token_tails.size() !=
            static_cast<std::size_t>(metadata.experts_per_gpu)) {
        throw std::runtime_error("expert token progress count does not match metadata");
    }
    for (int expert = 0; expert < metadata.experts_per_gpu; ++expert) {
        const auto expert_index = static_cast<std::size_t>(expert);
        const auto list_begin = static_cast<std::size_t>(
            metadata.expert_offsets[expert_index]);
        const auto list_end = static_cast<std::size_t>(
            metadata.expert_offsets[expert_index + 1]);
        auto& head = progress.expert_token_heads[expert_index];
        const auto old_head = head;
        while (list_begin + head < list_end &&
               static_cast<std::size_t>(
                   metadata.expert_token_indices[list_begin + head]) < received_end) {
            ++head;
        }
        scheduler.expert_num_ready_tokens[expert_index] += head - old_head;
        if (flush_ready_entries_per_entry) {
            schedule_ready_computation_batches_for_expert_locked(
                expert_index, /*flush_per_entry=*/true);
        }
    }
    progress.received_token_frontier = received_end;
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
    if (!is_router_computation_input(*it)) {
        throw std::runtime_error(
            "received a completion for a source buffer excluded from QuACK inputs");
    }
    update_expert_token_heads_locked(
        *it, iteration, start_token, num_tokens,
        /*flush_ready_entries_per_entry=*/false);
}

std::size_t
CudaBuffers::schedule_ready_computation_batches_for_expert_locked(
    std::size_t expert,
    bool flush_per_entry) {
    auto& scheduler = router_computation_scheduler_state_;
    auto& publication = router_notification_publication_buffers_;
    if (!scheduler.initialized || !scheduler.iteration_initialized) {
        throw std::runtime_error("router computation scheduler iteration is not initialized");
    }
    if (expert >= scheduler.expert_total_tokens.size()) {
        throw std::runtime_error("router computation expert index is out of range");
    }
    const auto first_new_row = scheduler.published_rows;
    const auto m_cluster_tokens =
        config_.expert_gemm_m_tile * config_.expert_gemm_cluster_m;
    while (scheduler.expert_num_notified_batches[expert] <
           scheduler.expert_total_batches[expert]) {
        const auto next_expert_batch =
            scheduler.expert_num_notified_batches[expert];
        const bool is_last = next_expert_batch + 1 ==
            scheduler.expert_total_batches[expert];
        const auto batch_tokens = is_last
            ? scheduler.expert_total_tokens[expert] -
                next_expert_batch * m_cluster_tokens
            : m_cluster_tokens;
        if (scheduler.expert_num_ready_tokens[expert] < batch_tokens) {
            break;
        }
        if (scheduler.published_rows > publication.table_rows ||
            publication.n_groups_per_m_cluster >
                publication.table_rows - scheduler.published_rows) {
            throw std::runtime_error("QuACK gather-table capacity exceeded");
        }

        std::size_t remaining = batch_tokens;
        std::vector<int32_t> ranges(
            2 * router_computation_buffer_indices_.size(), 0);
        for (std::size_t buffer_index = 0;
             buffer_index < router_computation_buffer_indices_.size();
             ++buffer_index) {
            auto& buffer = nvlink_recv_buffers_[
                router_computation_buffer_indices_[buffer_index]];
            auto& progress = buffer.expert_token_head_state;
            const auto head = progress.expert_token_heads[expert];
            auto& tail = progress.expert_token_tails[expert];
            if (tail > head) {
                throw std::runtime_error("expert token Tail exceeds Head");
            }
            const auto available = head - tail;
            const auto selected = std::min(available, remaining);
            const auto expert_offset = static_cast<std::size_t>(
                buffer.expert_metadata.expert_offsets[expert]);
            if (tail > std::numeric_limits<std::size_t>::max() - expert_offset) {
                throw std::runtime_error(
                    "QuACK gather-table route range overflows size_t");
            }
            const auto start = expert_offset + tail;
            if (selected > std::numeric_limits<std::size_t>::max() - start ||
                start + selected > static_cast<std::size_t>(
                    std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error(
                    "QuACK gather-table route range exceeds int32 ABI");
            }
            ranges[2 * buffer_index] = static_cast<int32_t>(start);
            ranges[2 * buffer_index + 1] = static_cast<int32_t>(start + selected);
            tail += selected;
            remaining -= selected;
        }
        if (remaining != 0) {
            throw std::runtime_error(
                "num_ready_tokens is inconsistent with expert Head/Tail state");
        }

        for (std::size_t n_group = 0;
             n_group < publication.n_groups_per_m_cluster; ++n_group) {
            const auto table_row = scheduler.published_rows;
            auto* row = reinterpret_cast<int32_t*>(
                static_cast<uint8_t*>(publication.host_table.ptr) +
                table_row * publication.table_row_bytes);
            row[0] = static_cast<int32_t>(expert);
            row[1] = static_cast<int32_t>(n_group) * publication.group_size;
            std::copy(ranges.begin(), ranges.end(), row + 2);
            ++scheduler.published_rows;
            *static_cast<int32_t*>(publication.host_ready_rows.ptr) =
                static_cast<int32_t>(scheduler.published_rows);
            if (flush_per_entry) {
                flush_router_notification_publication_range(
                    table_row * publication.table_row_bytes,
                    publication.table_row_bytes);
            }
        }
        scheduler.expert_num_ready_tokens[expert] -= batch_tokens;
        ++scheduler.expert_num_notified_batches[expert];
        ++scheduler.published_batches;
    }

    return scheduler.published_rows - first_new_row;
}

std::size_t CudaBuffers::schedule_ready_computation_batches_locked() {
    auto& scheduler = router_computation_scheduler_state_;
    const auto first_new_row = scheduler.published_rows;
    for (std::size_t expert = 0;
         expert < scheduler.expert_total_tokens.size(); ++expert) {
        schedule_ready_computation_batches_for_expert_locked(
            expert, /*flush_per_entry=*/false);
    }
    return scheduler.published_rows - first_new_row;
}

void CudaBuffers::process_router_notification_completion(
    int source_node_rank,
    int source_gpu_index,
    uint64_t iteration,
    std::size_t start_token,
    std::size_t num_tokens) {
    if (config_.nvlink_forward_notification_flush_only_enabled) {
        const auto row_bytes =
            router_notification_publication_buffers_.table_row_bytes;
        flush_router_notification_publication_range(0, row_bytes);
        return;
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
    const bool flush_per_entry =
        config_.nvlink_forward_notification_flush_per_entry_enabled;
    update_expert_token_heads_locked(
        *it, iteration, start_token, num_tokens,
        /*flush_ready_entries_per_entry=*/flush_per_entry);
    if (flush_per_entry) {
        return;
    }
    const auto first_new_row =
        router_computation_scheduler_state_.published_rows;
    const auto new_rows = schedule_ready_computation_batches_locked();
    if (new_rows != 0) {
        flush_router_notification_publication_range(
            first_new_row *
                router_notification_publication_buffers_.table_row_bytes,
            new_rows *
                router_notification_publication_buffers_.table_row_bytes);
    }
}

void CudaBuffers::flush_router_notification_publication() {
    flush_router_notification_publication_range(
        0, router_notification_publication_buffers_.host_table.bytes);
}

void CudaBuffers::flush_router_notification_publication_range(
    std::size_t table_offset,
    std::size_t table_bytes) {
    auto& publication = router_notification_publication_buffers_;
    if (!publication.host_ready_rows.ptr || !publication.device_ready_rows.ptr ||
        publication.host_ready_rows.bytes != sizeof(int32_t) ||
        publication.device_ready_rows.bytes != sizeof(int32_t)) {
        throw std::runtime_error(
            "router notification publication buffers are not initialized");
    }
    if (table_offset > publication.host_table.bytes ||
        table_bytes > publication.host_table.bytes - table_offset ||
        publication.host_table.bytes != publication.device_table.bytes ||
        (table_bytes != 0 &&
         (!publication.host_table.ptr || !publication.device_table.ptr))) {
        throw std::runtime_error("QuACK gather-table flush range is invalid");
    }

    const auto ready_rows = *static_cast<const int32_t*>(
        publication.host_ready_rows.ptr);
    if (ready_rows < 0 ||
        static_cast<std::size_t>(ready_rows) > publication.table_rows) {
        throw std::runtime_error(
            "QuACK ready-row publication value is outside the table range");
    }
    const void* ready_rows_memcpy_source = publication.host_ready_rows.ptr;
    if (config_.nvlink_forward_notification_flag_update_mode ==
        NvlinkForwardNotificationFlagUpdateMode::kMemcpy) {
        const auto required_ready_value_bytes =
            (publication.table_rows + 1) * sizeof(int32_t);
        if (!router_notification_ready_values_.ptr ||
            router_notification_ready_values_.bytes !=
                required_ready_value_bytes) {
            throw std::runtime_error(
                "NVLink notification memcpy flag staging is not initialized");
        }
        ready_rows_memcpy_source =
            static_cast<const int32_t*>(
                router_notification_ready_values_.ptr) + ready_rows;
    }

    if (config_.mock_mode) {
        if (table_bytes != 0) {
            std::lock_guard<std::mutex> timing_lock(
                router_computation_timing_mutex_);
            if (router_computation_timing_initialized_ &&
                !router_computation_start_recorded_) {
                router_computation_start_recorded_ = true;
            }
        }
        if (table_bytes != 0) {
            std::memcpy(
                static_cast<uint8_t*>(publication.device_table.ptr) + table_offset,
                static_cast<const uint8_t*>(publication.host_table.ptr) + table_offset,
                table_bytes);
        }
        std::memcpy(
            publication.device_ready_rows.ptr,
            ready_rows_memcpy_source,
            publication.host_ready_rows.bytes);
        if (table_bytes != 0) ++publication.table_flush_count;
        ++publication.ready_rows_publication_count;
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    if (!router_notification_publication_stream_) {
        throw std::runtime_error(
            "router notification publication stream is not initialized");
    }
    auto stream = reinterpret_cast<cudaStream_t>(
        router_notification_publication_stream_);
    if (table_bytes != 0) {
        {
            std::lock_guard<std::mutex> timing_lock(
                router_computation_timing_mutex_);
            if (!router_computation_timing_initialized_) {
                throw std::runtime_error(
                    "router computation timing iteration is not initialized");
            }
            if (!router_computation_start_recorded_) {
                check_cuda(
                    cudaEventRecord(
                        reinterpret_cast<cudaEvent_t>(
                            router_computation_start_event_),
                        stream),
                    "cudaEventRecord QuACK computation start");
                router_computation_start_recorded_ = true;
            }
        }
        check_cuda(cudaMemcpyAsync(
            static_cast<uint8_t*>(publication.device_table.ptr) + table_offset,
            static_cast<const uint8_t*>(publication.host_table.ptr) + table_offset,
            table_bytes,
            cudaMemcpyHostToDevice,
            stream), "cudaMemcpyAsync H2D QuACK gather-table range");
    }

    if (config_.nvlink_forward_notification_flag_update_mode ==
        NvlinkForwardNotificationFlagUpdateMode::kMemcpy) {
        check_cuda(cudaMemcpyAsync(
            publication.device_ready_rows.ptr,
            ready_rows_memcpy_source,
            sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream),
            "cudaMemcpyAsync H2D QuACK gather ready rows");
    } else {
        // A stream memory write is host-asynchronous, writes the aligned flag
        // as one 32-bit value, and (without NO_MEMORY_BARRIER) performs a
        // system-wide fence after the preceding table copy and before
        // publishing ready rows.
        check_cuda_driver(cuStreamWriteValue32(
            reinterpret_cast<CUstream>(stream),
            reinterpret_cast<CUdeviceptr>(publication.device_ready_rows.ptr),
            static_cast<cuuint32_t>(ready_rows),
            CU_STREAM_WRITE_VALUE_DEFAULT),
            "cuStreamWriteValue32 QuACK gather ready rows");
    }
    if (table_bytes != 0) ++publication.table_flush_count;
    ++publication.ready_rows_publication_count;
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

void CudaBuffers::allocate_buffer(
    GpuBuffer& buffer,
    const DeviceBufferAllocationRequest& request) {
    if (request.bytes == 0) {
        throw std::runtime_error("cannot allocate a zero-byte device buffer");
    }
    buffer.bytes = request.bytes;
    buffer.is_mock_host_memory = config_.mock_mode;
    if (external_device_buffer_allocator_) {
        buffer.ptr = external_device_buffer_allocator_(request);
        if (!buffer.ptr) {
            buffer.bytes = 0;
            throw std::runtime_error("external device-buffer allocator returned a null pointer");
        }
        buffer.is_externally_owned = true;
        RDMA_PROXY_LOG_INFO(
            "borrowed externally allocated device buffer kind=",
            static_cast<int>(request.kind),
            " bytes=", request.bytes,
            " peer=", request.peer_rank,
            " source_node=", request.source_node_rank,
            " source_gpu=", request.source_gpu_index);
        return;
    }
    if (config_.mock_mode) {
        buffer.ptr = ::operator new(request.bytes);
        std::memset(buffer.ptr, 0, request.bytes);
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaMalloc(&buffer.ptr, request.bytes), "cudaMalloc");
    check_cuda(cudaMemset(buffer.ptr, 0, request.bytes), "cudaMemset");
#else
    throw std::runtime_error("CUDA allocation requested but CUDA support was not built");
#endif
}

void CudaBuffers::alias_buffer(
    GpuBuffer& buffer,
    const GpuBuffer& source,
    const DeviceBufferAllocationRequest& request) {
    if (!source.ptr || source.bytes != request.bytes) {
        throw std::runtime_error(
            "device-buffer alias source is missing or has the wrong size");
    }
    if (external_device_buffer_allocator_) {
        void* exposed_ptr = external_device_buffer_allocator_(request);
        if (!exposed_ptr) {
            throw std::runtime_error(
                "external device-buffer allocator rejected an alias request");
        }
        if (exposed_ptr != source.ptr) {
            throw std::runtime_error(
                "external device-buffer allocator did not return the RDMA receive "
                "pointer for a direct-input alias");
        }
    }
    buffer = source;
    buffer.is_alias = true;
}

void CudaBuffers::free_buffer(GpuBuffer& buffer) {
    if (!buffer.ptr) return;
    if (buffer.is_alias) {
        buffer = GpuBuffer{};
        return;
    }
    if (buffer.is_externally_owned) {
        buffer = GpuBuffer{};
        return;
    }
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
    buffer = GpuBuffer{};
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

void select_cuda_device_for_thread(int cuda_device_id, bool mock_mode) {
    if (mock_mode) return;
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaSetDevice(cuda_device_id), "cudaSetDevice for proxy worker thread");
#else
    (void)cuda_device_id;
    throw std::runtime_error("CUDA device selection requested without CUDA support");
#endif
}

void flush_gpudirect_rdma_writes(int cuda_device_id, bool mock_mode) {
    if (mock_mode) return;
#if RDMA_PROXY_HAVE_CUDA
    select_cuda_device_for_thread(cuda_device_id, mock_mode);
    check_cuda_driver(
        cuFlushGPUDirectRDMAWrites(
            CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
            CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER),
        "cuFlushGPUDirectRDMAWrites direct input visibility");
#else
    (void)cuda_device_id;
    throw std::runtime_error("GPUDirect RDMA visibility flush requested without CUDA support");
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

CudaIpcMemoryHandle export_cuda_ipc_memory_handle(void* ptr, bool mock_mode) {
    if (!ptr) throw std::runtime_error("cannot export null CUDA IPC pointer");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    if (mock_mode) {
        out << std::setw(sizeof(uintptr_t) * 2) << reinterpret_cast<uintptr_t>(ptr);
        return CudaIpcMemoryHandle{out.str(), 0};
    }
#if RDMA_PROXY_HAVE_CUDA
    CUdeviceptr allocation_base = 0;
    std::size_t allocation_bytes = 0;
    check_cuda_driver(
        cuMemGetAddressRange(
            &allocation_base,
            &allocation_bytes,
            reinterpret_cast<CUdeviceptr>(ptr)),
        "cuMemGetAddressRange");
    const auto pointer_value = reinterpret_cast<uintptr_t>(ptr);
    const auto base_value = static_cast<uintptr_t>(allocation_base);
    if (pointer_value < base_value ||
        pointer_value - base_value >= allocation_bytes) {
        throw std::runtime_error("CUDA IPC pointer is outside its allocation range");
    }
    cudaIpcMemHandle_t handle{};
    check_cuda(
        cudaIpcGetMemHandle(
            &handle, reinterpret_cast<void*>(allocation_base)),
        "cudaIpcGetMemHandle");
    const auto* bytes = reinterpret_cast<const unsigned char*>(&handle);
    for (std::size_t i = 0; i < sizeof(handle); ++i) {
        out << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return CudaIpcMemoryHandle{out.str(), pointer_value - base_value};
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
