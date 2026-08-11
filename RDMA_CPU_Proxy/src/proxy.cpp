#include "proxy.hpp"

#include "logging.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rdma_proxy {

struct alignas(64) Proxy::LocalIterationSyncHeader {
    uint64_t magic{0};
    uint32_t version{0};
    uint32_t initialized{0};
    int32_t node_rank{-1};
    uint32_t num_gpus_per_node{0};
    uint32_t header_bytes{0};
    uint32_t slot_bytes{0};
    uint32_t reserved{0};
    char padding[28]{};
};

struct alignas(64) Proxy::LocalIterationSyncSlot {
    int32_t pid{0};
    int32_t gpu_index{-1};
    uint64_t iteration_start{0};
    uint64_t iteration_done{0};
    uint64_t nvlink_forward_batch_start{0};
    uint64_t nvlink_forward_batch_chunks{0};
    uint64_t nvlink_forward_batch_selected{0};
    uint64_t nvlink_forward_batch_selected_chunks{0};
    char padding[8]{};
};

struct alignas(64) Proxy::NvlinkForwardNotification {
    uint64_t sequence{0};
    uint64_t iteration{0};
    uint64_t batch_index{0};
    uint64_t peer_slot{0};
    uint64_t start_token{0};
    uint64_t num_tokens{0};
    uint64_t byte_offset{0};
    uint64_t bytes{0};
    int32_t source_gpu{-1};
    int32_t destination_gpu{-1};
    int32_t peer_rank{-1};
    uint32_t flags{0};
    char padding[12]{};
};

struct alignas(64) Proxy::NvlinkForwardNotificationQueue {
    int32_t source_gpu{-1};
    uint32_t capacity{0};
    uint64_t head{0};
    uint64_t tail{0};
    uint64_t dropped{0};
    uint64_t sender_done{0};
    char padding[24]{};
};

struct alignas(64) Proxy::NvlinkForwardNotificationHeader {
    uint64_t magic{0};
    uint32_t version{0};
    uint32_t initialized{0};
    int32_t node_rank{-1};
    int32_t gpu_index{-1};
    int32_t pid{0};
    uint32_t num_gpus_per_node{0};
    uint32_t queue_count{0};
    uint32_t queue_capacity{0};
    uint32_t header_bytes{0};
    uint32_t queue_bytes{0};
    uint32_t notification_bytes{0};
    uint32_t entry_bytes{0};
    uint32_t reserved{0};
};

struct Proxy::NvlinkForwardNotificationDispatchState {
    std::mutex mutex;
    std::deque<NvlinkForwardNotification> pending;
};

namespace {

constexpr uint64_t kLocalIterationSyncMagic = 0x52444d415053594eULL;  // "RDMAPSyn"
constexpr uint32_t kLocalIterationSyncVersion = 3;
constexpr uint64_t kNvlinkForwardNotificationMagic = 0x52444d414e464e51ULL;  // "RDMANFNQ"
constexpr uint32_t kNvlinkForwardNotificationVersion = 2;
constexpr uint32_t kNvlinkForwardRoundRobinFlag = 1U;

std::size_t checked_add(std::size_t a, std::size_t b, const char* description) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::runtime_error(std::string(description) + " overflows size_t");
    }
    return a + b;
}

std::size_t checked_multiply(std::size_t a, std::size_t b, const char* description) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string(description) + " overflows size_t");
    }
    return a * b;
}

uint32_t atomic_load_u32(const uint32_t* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

void atomic_store_u32(uint32_t* ptr, uint32_t value) {
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

int32_t atomic_load_i32(const int32_t* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

void atomic_store_i32(int32_t* ptr, int32_t value) {
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

uint64_t atomic_load_u64(const uint64_t* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

void atomic_store_u64(uint64_t* ptr, uint64_t value) {
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

uint64_t atomic_fetch_add_u64(uint64_t* ptr, uint64_t value) {
    return __atomic_fetch_add(ptr, value, __ATOMIC_ACQ_REL);
}

void atomic_thread_fence_release() {
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

void atomic_thread_fence_acquire() {
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

int current_process_id() {
#if defined(__unix__) || defined(__APPLE__)
    return static_cast<int>(getpid());
#else
    return 0;
#endif
}

bool process_is_alive(int pid) {
    if (pid <= 0) return false;
#if defined(__unix__) || defined(__APPLE__)
    return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
#else
    return true;
#endif
}

std::string sanitize_path_component(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (const char c : value) {
        const auto uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '.' || c == '_' || c == '-') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "default" : out;
}

std::string default_local_iteration_sync_run_id(const ProxyConfig& config) {
    std::ostringstream out;
    out << "nodes_" << config.num_nodes
        << "_gpus_" << config.num_gpus_per_node;
    return out.str();
}

uint64_t fnv1a64(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (const auto c : value) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint8_t routing_column_mask(std::size_t column) {
    return static_cast<uint8_t>(1U << (7U - column));
}

std::size_t active_routing_columns(int num_gpus_per_node) {
    return static_cast<std::size_t>(num_gpus_per_node - 1);
}

int routing_column_to_gpu(int local_gpu, std::size_t column, int num_gpus_per_node) {
    if (column >= active_routing_columns(num_gpus_per_node)) return -1;
    return (local_gpu + 1 + static_cast<int>(column)) % num_gpus_per_node;
}

int64_t elapsed_us_since(
    std::chrono::steady_clock::time_point time,
    std::chrono::steady_clock::time_point start) {
    if (time == std::chrono::steady_clock::time_point{}) return -1;
    return static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(time - start).count());
}

int64_t nonnegative_delta_us(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
    if (start == std::chrono::steady_clock::time_point{} ||
        end == std::chrono::steady_clock::time_point{}) {
        return -1;
    }
    const auto delta = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    return std::max<int64_t>(0, delta);
}

std::size_t routing_column_for_gpu(int local_gpu, int dst_gpu, int num_gpus_per_node) {
    const auto column = static_cast<std::size_t>(
        (dst_gpu - local_gpu - 1 + num_gpus_per_node) % num_gpus_per_node);
    if (column >= active_routing_columns(num_gpus_per_node)) {
        throw std::runtime_error("NVLink routing destination maps outside active routing columns");
    }
    return column;
}

std::size_t nvlink_notification_queue_index_for_source(int receiver_gpu, int source_gpu) {
    if (source_gpu == receiver_gpu) {
        throw std::runtime_error("NVLink notification source GPU cannot match receiver GPU");
    }
    return static_cast<std::size_t>(source_gpu < receiver_gpu ? source_gpu : source_gpu - 1);
}

uint64_t unix_epoch_nanoseconds_now() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

}  // namespace

Proxy::Proxy(ProxyConfig config)
    : config_(std::move(config)),
      cuda_buffers_(config_),
      router_routing_(config_),
      rdma_context_(config_),
      connection_manager_(config_) {}

Proxy::~Proxy() {
    shutdown();
}

void Proxy::initialize() {
    if (initialized_) return;
    RDMA_PROXY_LOG_INFO("initializing proxy: ", config_summary(config_));

    rdma_iteration_bandwidth_gbps_.clear();
    initialize_local_iteration_sync();
    router_routing_.initialize();
    cuda_buffers_.initialize();
    exchange_router_expert_metadata_all_to_all();
    rdma_context_.initialize();

    for (auto& peer_buffers : cuda_buffers_.peer_buffers()) {
        setup_peer(peer_buffers);
    }
    start_forwarding_thread();
    initialized_ = true;
}

const RouterExpertMetadata& Proxy::router_expert_metadata_for_source(
    int source_node_rank,
    int source_gpu_index) const {
    const auto& buffer = cuda_buffers_.nvlink_receive_buffer_for_source(
        source_node_rank, source_gpu_index);
    if (!buffer.expert_metadata_ready) {
        throw std::runtime_error("router expert metadata is not ready for source GPU");
    }
    return buffer.expert_metadata;
}

void Proxy::run_once() {
    if (!initialized_) throw std::runtime_error("proxy is not initialized");
    run_iteration(0);
    synchronize_iteration(0);
    report_rdma_bandwidth_summary();
}

void Proxy::run() {
    if (!initialized_) throw std::runtime_error("proxy is not initialized");
    for (uint64_t iteration = 0;
         config_.num_iterations == 0 || iteration < static_cast<uint64_t>(config_.num_iterations);
         ++iteration) {
        run_iteration(iteration);
        synchronize_iteration(iteration);
    }
    if (config_.num_iterations != 0) {
        report_rdma_bandwidth_summary();
    }
}

void Proxy::shutdown() {
    stop_forwarding_thread();
    for (auto& peer : peers_) {
        for (auto& worker : peer.workers) {
            if (worker) worker->stop();
        }
        rdma_context_.deregister_memory(peer.local_send_mr);
        rdma_context_.deregister_memory(peer.local_recv_mr);
    }
    peers_.clear();
    release_local_iteration_sync();
    initialized_ = false;
}

PeerConnectionInfo Proxy::make_local_peer_info(const PeerState& peer) const {
    PeerConnectionInfo info;
    info.node_rank = config_.node_rank;
    info.gpu_index = config_.local_gpu_index;
    info.recv_buffer = peer.local_recv_mr;
    info.qps.reserve(peer.qps.size());
    for (const auto& qp : peer.qps) {
        info.qps.push_back(qp->local_info());
    }
    return info;
}

void Proxy::setup_peer(PeerGpuBuffers& buffers) {
    PeerState peer;
    peer.peer_rank = buffers.peer_rank;
    peer.local_send_mr = rdma_context_.register_memory(
        buffers.send.ptr, buffers.send.bytes, "send_buffer_peer_" + std::to_string(buffers.peer_rank));
    peer.local_recv_mr = rdma_context_.register_memory(
        buffers.recv.ptr, buffers.recv.bytes, "recv_buffer_peer_" + std::to_string(buffers.peer_rank));

    for (int q = 0; q < config_.num_qps_per_peer; ++q) {
        peer.qps.emplace_back(new RdmaQueuePair(rdma_context_, config_, buffers.peer_rank, q));
    }

    const auto local_info = make_local_peer_info(peer);
    const auto& peer_addr = *std::find_if(config_.peers.begin(), config_.peers.end(), [&](const PeerAddress& p) {
        return p.node_rank == buffers.peer_rank;
    });
    const auto remote_info = connection_manager_.exchange_peer_info(peer_addr, local_info);
    if (remote_info.qps.size() != peer.qps.size()) {
        throw std::runtime_error("remote QP count does not match local QP count");
    }
    peer.remote_recv_mr = remote_info.recv_buffer;
    peer.remote_gpu_index = remote_info.gpu_index;
    if (config_.router_routing_enabled) {
        auto remote_router = exchange_router_receive_metadata(peer_addr);
        peer.receive_chunks = compute_chunks_from_token_indices(
            remote_router.token_indices,
            config_.token_dimension,
            dtype_size(config_.dtype),
            config_.tokens_per_chunk,
            config_.num_qps_per_peer);
        peer.router_x4 = std::move(remote_router.token_masks);
    } else {
        peer.receive_chunks = make_chunks();
    }

    for (std::size_t q = 0; q < peer.qps.size(); ++q) {
        peer.qps[q]->connect(remote_info.qps[q]);
        auto worker = std::make_unique<QPWorker>(
            *peer.qps[q],
            config_.completion_poll_batch_size,
            config_.max_in_flight_chunks_per_qp);
        worker->configure_expected_chunks(peer.receive_chunks.size());
        if (config_.nvlink_forward_out_of_order_chunks_enabled) {
            const auto peer_index = peers_.size();
            worker->set_receive_immediate_callback(
                [this, peer_index](std::size_t chunk_index) {
                    record_out_of_order_forwarding_arrival(peer_index, chunk_index);
                });
        }
        worker->post_initial_receives(config_.recv_queue_depth);
        worker->start();
        peer.workers.emplace_back(std::move(worker));
    }

    synchronize_peer_ready(peer_addr, peer);
    RDMA_PROXY_LOG_INFO("peer ", buffers.peer_rank, " initialized with ", peer.qps.size(), " RC QPs");
    peers_.push_back(std::move(peer));
}

RouterX3Metadata Proxy::exchange_router_receive_metadata(
    const PeerAddress& peer_addr) const {
    RouterX3Metadata local;
    local.source_node_rank = config_.node_rank;
    local.destination_node_rank = peer_addr.node_rank;
    local.local_gpu_index = config_.local_gpu_index;
    local.num_nodes = config_.num_nodes;
    local.num_gpus_per_node = config_.num_gpus_per_node;
    local.num_experts = config_.router_num_experts;
    local.top_k = config_.router_top_k;
    local.num_tokens = config_.num_tokens;
    local.token_dimension = config_.token_dimension;
    local.element_bytes = dtype_size(config_.dtype);
    local.tokens_per_chunk = config_.tokens_per_chunk;
    local.token_indices = router_routing_.token_indices_for_node(peer_addr.node_rank);
    local.token_masks = router_routing_.token_masks_for_node(peer_addr.node_rank);

    const auto remote_payload = connection_manager_.exchange_control_message(
        peer_addr,
        serialize_router_x3_metadata(local),
        config_.completion_timeout_ms);
    const auto remote = deserialize_router_x3_metadata(remote_payload, config_.num_tokens);
    const int expected_source = config_.mock_mode ? config_.node_rank : peer_addr.node_rank;
    const int expected_destination = config_.mock_mode ? peer_addr.node_rank : config_.node_rank;
    if (remote.source_node_rank != expected_source ||
        remote.destination_node_rank != expected_destination ||
        remote.local_gpu_index != config_.local_gpu_index) {
        throw std::runtime_error("router x3/x4 TCP metadata rank/GPU mismatch");
    }
    if (remote.num_nodes != config_.num_nodes ||
        remote.num_gpus_per_node != config_.num_gpus_per_node ||
        remote.num_experts != config_.router_num_experts ||
        remote.top_k != config_.router_top_k ||
        remote.num_tokens != config_.num_tokens ||
        remote.token_dimension != config_.token_dimension ||
        remote.element_bytes != dtype_size(config_.dtype) ||
        remote.tokens_per_chunk != config_.tokens_per_chunk) {
        throw std::runtime_error("router x3/x4 TCP metadata tensor/chunk configuration mismatch");
    }
    RDMA_PROXY_LOG_INFO(
        "received router x3/x4 metadata peer=", peer_addr.node_rank,
        " routed_tokens=", remote.token_indices.size(),
        " chunks=", (remote.token_indices.size() + config_.tokens_per_chunk - 1) /
            config_.tokens_per_chunk);
    return remote;
}

void Proxy::exchange_router_expert_metadata_all_to_all() {
    if (!config_.router_routing_enabled || !config_.nvlink_forwarding_enabled) return;

    const auto maximum_index_count = config_.num_tokens *
        static_cast<std::size_t>(config_.router_top_k);
    const auto install = [&](RouterExpertMetadata metadata) {
        if (metadata.destination_node_rank != config_.node_rank ||
            metadata.destination_gpu_index != config_.local_gpu_index ||
            metadata.num_nodes != config_.num_nodes ||
            metadata.num_gpus_per_node != config_.num_gpus_per_node ||
            metadata.num_experts != config_.router_num_experts ||
            metadata.experts_per_gpu != router_routing_.experts_per_gpu() ||
            metadata.num_tokens != config_.num_tokens) {
            throw std::runtime_error(
                "received router expert metadata does not match local topology");
        }
        cuda_buffers_.install_expert_metadata(std::move(metadata));
    };

    if (config_.mock_mode) {
        // A one-process mock run has no remote proxies. Recreate their
        // deterministic router state so every source-specific buffer still
        // receives semantically valid expert metadata.
        for (int source_node = 0; source_node < config_.num_nodes; ++source_node) {
            for (int source_gpu = 0; source_gpu < config_.num_gpus_per_node; ++source_gpu) {
                if (source_node == config_.node_rank &&
                    source_gpu == config_.local_gpu_index) {
                    install(router_routing_.expert_metadata_for_gpu(
                        config_.node_rank, config_.local_gpu_index));
                    continue;
                }
                auto source_config = config_;
                source_config.node_rank = source_node;
                source_config.local_gpu_index = source_gpu;
                RouterRouting source_routing(std::move(source_config));
                source_routing.initialize();
                install(source_routing.expert_metadata_for_gpu(
                    config_.node_rank, config_.local_gpu_index));
            }
        }
        return;
    }

    const uint16_t local_metadata_port = static_cast<uint16_t>(
        static_cast<unsigned>(config_.router_metadata_port_base) +
        static_cast<unsigned>(config_.local_gpu_index));
    install(router_routing_.expert_metadata_for_gpu(
        config_.node_rank, config_.local_gpu_index));

    struct OutgoingExpertMetadata {
        PeerAddress endpoint;
        std::string payload;
    };
    std::vector<OutgoingExpertMetadata> outgoing;
    const auto total_gpus = static_cast<std::size_t>(
        config_.num_nodes * config_.num_gpus_per_node);
    outgoing.reserve(total_gpus - 1);
    for (int peer_node = 0; peer_node < config_.num_nodes; ++peer_node) {
        std::string peer_host = "::1";
        if (peer_node != config_.node_rank) {
            const auto node_peer = std::find_if(
                config_.peers.begin(), config_.peers.end(),
                [&](const PeerAddress& candidate) {
                    return candidate.node_rank == peer_node;
                });
            if (node_peer == config_.peers.end()) {
                throw std::runtime_error(
                    "missing node host for router expert metadata exchange");
            }
            peer_host = node_peer->host;
        }
        for (int peer_gpu = 0; peer_gpu < config_.num_gpus_per_node; ++peer_gpu) {
            if (peer_node == config_.node_rank &&
                peer_gpu == config_.local_gpu_index) {
                continue;
            }

            OutgoingExpertMetadata message;
            message.endpoint.node_rank = peer_node;
            message.endpoint.host = peer_host;
            message.endpoint.port = static_cast<uint16_t>(
                static_cast<unsigned>(config_.router_metadata_port_base) +
                static_cast<unsigned>(peer_gpu));
            message.payload = serialize_router_expert_metadata(
                router_routing_.expert_metadata_for_gpu(peer_node, peer_gpu));
            outgoing.push_back(std::move(message));
        }
    }

    if (maximum_index_count >
        (std::numeric_limits<std::size_t>::max() - 4096) / 16) {
        throw std::runtime_error("router expert metadata payload bound overflows size_t");
    }
    const auto maximum_payload_bytes = std::size_t{4096} +
        (maximum_index_count +
         static_cast<std::size_t>(router_routing_.experts_per_gpu() + 1)) * 16;
    std::vector<std::string> received_payloads;
    std::exception_ptr receive_error;
    std::thread receiver([&] {
        try {
            received_payloads = connection_manager_.receive_global_control_messages(
                local_metadata_port,
                total_gpus - 1,
                maximum_payload_bytes,
                config_.completion_timeout_ms);
        } catch (...) {
            receive_error = std::current_exception();
        }
    });

    std::exception_ptr send_error;
    for (const auto& message : outgoing) {
        try {
            connection_manager_.send_global_control_message(
                message.endpoint, message.payload, config_.completion_timeout_ms);
        } catch (...) {
            if (!send_error) send_error = std::current_exception();
        }
    }
    receiver.join();
    if (send_error) std::rethrow_exception(send_error);
    if (receive_error) std::rethrow_exception(receive_error);

    for (const auto& remote_payload : received_payloads) {
        auto remote = deserialize_router_expert_metadata(
            remote_payload, config_.num_tokens, maximum_index_count);
        if (remote.source_node_rank < 0 ||
            remote.source_node_rank >= config_.num_nodes ||
            remote.source_gpu_index < 0 ||
            remote.source_gpu_index >= config_.num_gpus_per_node ||
            remote.destination_node_rank != config_.node_rank ||
            remote.destination_gpu_index != config_.local_gpu_index ||
            (remote.source_node_rank == config_.node_rank &&
             remote.source_gpu_index == config_.local_gpu_index)) {
            throw std::runtime_error(
                "router expert metadata source/destination identity mismatch");
        }
        RDMA_PROXY_LOG_INFO(
            "received router expert metadata source_node=", remote.source_node_rank,
            " source_gpu=", remote.source_gpu_index,
            " destination_node=", config_.node_rank,
            " destination_gpu=", config_.local_gpu_index,
            " experts=", remote.experts_per_gpu,
            " indices=", remote.expert_token_indices.size());
        install(std::move(remote));
    }
}

void Proxy::synchronize_peer_ready(const PeerAddress& peer_addr, const PeerState& peer) const {
    std::ostringstream local;
    local << "peer_ready"
          << " rank=" << config_.node_rank
          << " gpu=" << config_.local_gpu_index
          << " peer_rank=" << peer.peer_rank
          << " remote_gpu=" << peer.remote_gpu_index;
    const auto expected_rank = "rank=" + std::to_string(peer.peer_rank);
    const auto expected_gpu = "gpu=" + std::to_string(peer.remote_gpu_index);

    RDMA_PROXY_LOG_INFO("local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " waiting for peer ready remote_rank=", peer.peer_rank,
                        " remote_gpu=", peer.remote_gpu_index);
    const auto remote = connection_manager_.exchange_control_message(
        peer_addr, local.str(), config_.completion_timeout_ms);
    if (remote.find(expected_rank) == std::string::npos ||
        remote.find(expected_gpu) == std::string::npos) {
        throw std::runtime_error("peer ready synchronization mismatch: " + remote);
    }
    RDMA_PROXY_LOG_INFO("local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " peer ready remote_rank=", peer.peer_rank,
                        " remote_gpu=", peer.remote_gpu_index);
}

std::string Proxy::local_iteration_sync_shm_name() const {
    const auto run_id = config_.local_iteration_sync_run_id.empty() ?
        default_local_iteration_sync_run_id(config_) : config_.local_iteration_sync_run_id;
    std::ostringstream material;
    material << run_id
             << "|node=" << config_.node_rank
             << "|num_nodes=" << config_.num_nodes
             << "|gpus=" << config_.num_gpus_per_node;
    std::ostringstream out;
    out << "/rdma_lis_" << std::hex << fnv1a64(material.str());
    return out.str();
}

Proxy::LocalIterationSyncSlot* Proxy::local_iteration_sync_slot(int gpu_index) const {
    if (!local_iteration_sync_header_) return nullptr;
    if (gpu_index < 0 || gpu_index >= config_.num_gpus_per_node) {
        throw std::runtime_error("local iteration sync GPU index out of range");
    }
    auto* base = reinterpret_cast<char*>(local_iteration_sync_header_);
    return reinterpret_cast<LocalIterationSyncSlot*>(
        base + sizeof(LocalIterationSyncHeader) +
        static_cast<std::size_t>(gpu_index) * sizeof(LocalIterationSyncSlot));
}

void Proxy::initialize_local_iteration_sync() {
    const bool sync_required =
        config_.local_iteration_sync_enabled || config_.nvlink_forward_local_batch_sync_enabled;
    if (!sync_required || config_.num_gpus_per_node <= 1) return;
    if (local_iteration_sync_header_) return;

#if defined(__unix__) || defined(__APPLE__)
    local_iteration_sync_name_ = local_iteration_sync_shm_name();
    local_iteration_sync_size_ = sizeof(LocalIterationSyncHeader) +
        static_cast<std::size_t>(config_.num_gpus_per_node) * sizeof(LocalIterationSyncSlot);

    bool created = false;
    int fd = shm_open(local_iteration_sync_name_.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) {
        created = true;
        if (ftruncate(fd, static_cast<off_t>(local_iteration_sync_size_)) != 0) {
            const auto error = errno;
            close(fd);
            throw std::runtime_error("failed to size local iteration shared memory " +
                                     local_iteration_sync_name_ + ": errno=" + std::to_string(error));
        }
    } else if (errno == EEXIST) {
        fd = shm_open(local_iteration_sync_name_.c_str(), O_RDWR, 0600);
        if (fd < 0) {
            throw std::runtime_error("failed to open local iteration shared memory " +
                                     local_iteration_sync_name_ + ": errno=" + std::to_string(errno));
        }
    } else {
        throw std::runtime_error("failed to create local iteration shared memory " +
                                 local_iteration_sync_name_ + ": errno=" + std::to_string(errno));
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    while (!created) {
        struct stat st {};
        if (fstat(fd, &st) != 0) {
            const auto error = errno;
            close(fd);
            throw std::runtime_error("failed to stat local iteration shared memory " +
                                     local_iteration_sync_name_ + ": errno=" + std::to_string(error));
        }
        if (st.st_size >= static_cast<off_t>(local_iteration_sync_size_)) break;
        if (std::chrono::steady_clock::now() >= deadline) {
            close(fd);
            throw std::runtime_error("timed out waiting for local iteration shared memory sizing: " +
                                     local_iteration_sync_name_);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    void* mapping = mmap(
        nullptr,
        local_iteration_sync_size_,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        0);
    if (mapping == MAP_FAILED) {
        const auto error = errno;
        close(fd);
        throw std::runtime_error("failed to map local iteration shared memory " +
                                 local_iteration_sync_name_ + ": errno=" + std::to_string(error));
    }

    auto* header = reinterpret_cast<LocalIterationSyncHeader*>(mapping);
    if (created) {
        std::memset(mapping, 0, local_iteration_sync_size_);
        header->magic = kLocalIterationSyncMagic;
        header->version = kLocalIterationSyncVersion;
        header->node_rank = config_.node_rank;
        header->num_gpus_per_node = static_cast<uint32_t>(config_.num_gpus_per_node);
        header->header_bytes = static_cast<uint32_t>(sizeof(LocalIterationSyncHeader));
        header->slot_bytes = static_cast<uint32_t>(sizeof(LocalIterationSyncSlot));
        atomic_store_u32(&header->initialized, 1);
    } else {
        while (atomic_load_u32(&header->initialized) != 1) {
            if (std::chrono::steady_clock::now() >= deadline) {
                munmap(mapping, local_iteration_sync_size_);
                close(fd);
                throw std::runtime_error("timed out waiting for local iteration shared memory initialization: " +
                                         local_iteration_sync_name_);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (header->magic != kLocalIterationSyncMagic ||
            header->version != kLocalIterationSyncVersion ||
            header->node_rank != config_.node_rank ||
            header->num_gpus_per_node != static_cast<uint32_t>(config_.num_gpus_per_node) ||
            header->header_bytes != sizeof(LocalIterationSyncHeader) ||
            header->slot_bytes != sizeof(LocalIterationSyncSlot)) {
            munmap(mapping, local_iteration_sync_size_);
            close(fd);
            throw std::runtime_error("local iteration shared memory metadata mismatch for " +
                                     local_iteration_sync_name_ +
                                     "; use a unique local_iteration_sync_run_id for this launch");
        }
    }

    local_iteration_sync_fd_ = fd;
    local_iteration_sync_header_ = header;

    auto* slot = local_iteration_sync_slot(config_.local_gpu_index);
    atomic_store_i32(&slot->gpu_index, config_.local_gpu_index);
    atomic_store_u64(&slot->iteration_start, 0);
    atomic_store_u64(&slot->iteration_done, 0);
    atomic_store_u64(&slot->nvlink_forward_batch_start, 0);
    atomic_store_u64(&slot->nvlink_forward_batch_chunks, 0);
    atomic_store_u64(&slot->nvlink_forward_batch_selected, 0);
    atomic_store_u64(&slot->nvlink_forward_batch_selected_chunks, 0);
    atomic_store_i32(&slot->pid, current_process_id());

    RDMA_PROXY_LOG_DEBUG("mapped local iteration shared memory local_rank=", config_.node_rank,
                         " local_gpu=", config_.local_gpu_index,
                         " name=", local_iteration_sync_name_,
                         " bytes=", local_iteration_sync_size_,
                         " created=", created ? "true" : "false");
#else
    throw std::runtime_error("local iteration shared-memory synchronization requires POSIX shared memory");
#endif
}

void Proxy::release_local_iteration_sync() {
#if defined(__unix__) || defined(__APPLE__)
    if (local_iteration_sync_header_) {
        munmap(local_iteration_sync_header_, local_iteration_sync_size_);
        local_iteration_sync_header_ = nullptr;
    }
    if (local_iteration_sync_fd_ >= 0) {
        close(local_iteration_sync_fd_);
        local_iteration_sync_fd_ = -1;
    }
    if (!local_iteration_sync_name_.empty()) {
        shm_unlink(local_iteration_sync_name_.c_str());
    }
    local_iteration_sync_size_ = 0;
    local_iteration_sync_name_.clear();
#else
    local_iteration_sync_header_ = nullptr;
    local_iteration_sync_size_ = 0;
    local_iteration_sync_fd_ = -1;
    local_iteration_sync_name_.clear();
#endif
}

std::string Proxy::nvlink_forward_notification_shm_name(int gpu_index) const {
    const auto run_id = config_.local_iteration_sync_run_id.empty() ?
        default_local_iteration_sync_run_id(config_) : config_.local_iteration_sync_run_id;
    std::ostringstream material;
    material << run_id
             << "|node=" << config_.node_rank
             << "|num_nodes=" << config_.num_nodes
             << "|gpus=" << config_.num_gpus_per_node
             << "|nvlink_notifications_gpu=" << gpu_index;
    std::ostringstream out;
    out << "/rdma_nfn_" << std::hex << fnv1a64(material.str());
    return out.str();
}

Proxy::NvlinkForwardNotificationQueue* Proxy::nvlink_forward_notification_queue_for_source(
    NvlinkForwardNotificationHeader* header,
    int source_gpu) const {
    if (!header) throw std::runtime_error("NVLink notification shared memory is not mapped");
    if (source_gpu < 0 || source_gpu >= static_cast<int>(header->num_gpus_per_node)) {
        throw std::runtime_error("NVLink notification source GPU out of range");
    }
    const auto queue_index = nvlink_notification_queue_index_for_source(header->gpu_index, source_gpu);
    if (queue_index >= header->queue_count) {
        throw std::runtime_error("NVLink notification queue index out of range");
    }
    auto* base = reinterpret_cast<char*>(header);
    auto* queues = reinterpret_cast<NvlinkForwardNotificationQueue*>(base + header->header_bytes);
    auto* queue = reinterpret_cast<NvlinkForwardNotificationQueue*>(
        reinterpret_cast<char*>(queues) + queue_index * header->queue_bytes);
    if (queue->source_gpu != source_gpu) {
        throw std::runtime_error("NVLink notification queue source GPU metadata mismatch");
    }
    return queue;
}

Proxy::NvlinkForwardNotification* Proxy::nvlink_forward_notification_entry(
    NvlinkForwardNotificationHeader* header,
    NvlinkForwardNotificationQueue* queue,
    uint64_t position) const {
    if (!header || !queue) throw std::runtime_error("NVLink notification queue is not mapped");
    if (queue->capacity == 0 || header->entry_bytes < sizeof(NvlinkForwardNotification)) {
        throw std::runtime_error("NVLink notification entry metadata is invalid");
    }
    const auto queue_index = nvlink_notification_queue_index_for_source(header->gpu_index, queue->source_gpu);
    auto* base = reinterpret_cast<char*>(header);
    const auto queues_bytes =
        static_cast<std::size_t>(header->queue_count) * static_cast<std::size_t>(header->queue_bytes);
    const auto entries_offset = static_cast<std::size_t>(header->header_bytes) + queues_bytes +
        queue_index * static_cast<std::size_t>(header->queue_capacity) *
            static_cast<std::size_t>(header->entry_bytes);
    const auto entry_offset = (position % queue->capacity) * static_cast<std::size_t>(header->entry_bytes);
    return reinterpret_cast<NvlinkForwardNotification*>(base + entries_offset + entry_offset);
}

void Proxy::initialize_nvlink_forward_notifications() {
    if (!config_.nvlink_forward_completion_notifications_enabled || config_.num_gpus_per_node <= 1) return;
    if (nvlink_forward_notification_header_) return;

#if defined(__unix__) || defined(__APPLE__)
    const auto queue_count = static_cast<std::size_t>(config_.num_gpus_per_node - 1);
    const auto queue_capacity = config_.nvlink_forward_notification_queue_depth;
    const auto entry_bytes = sizeof(NvlinkForwardNotification);
    if (entry_bytes > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("NVLink notification entry size exceeds uint32 range");
    }
    const auto queue_header_bytes = checked_multiply(
        queue_count, sizeof(NvlinkForwardNotificationQueue), "NVLink notification queue headers");
    const auto queue_entry_bytes = checked_multiply(
        checked_multiply(queue_count, queue_capacity, "NVLink notification queue slots"),
        entry_bytes,
        "NVLink notification queue entry storage");
    nvlink_forward_notification_size_ = checked_add(
        checked_add(
            sizeof(NvlinkForwardNotificationHeader),
            queue_header_bytes,
            "NVLink notification shared memory header size"),
        queue_entry_bytes,
        "NVLink notification shared memory size");
    nvlink_forward_notification_name_ = nvlink_forward_notification_shm_name(config_.local_gpu_index);

    int fd = shm_open(nvlink_forward_notification_name_.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0 && errno == EEXIST) {
        int stale_fd = shm_open(nvlink_forward_notification_name_.c_str(), O_RDWR, 0600);
        if (stale_fd >= 0) {
            struct stat st {};
            if (fstat(stale_fd, &st) == 0 &&
                st.st_size >= static_cast<off_t>(nvlink_forward_notification_size_)) {
                void* stale_mapping = mmap(
                    nullptr,
                    nvlink_forward_notification_size_,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    stale_fd,
                    0);
                if (stale_mapping != MAP_FAILED) {
                    auto* stale_header = reinterpret_cast<NvlinkForwardNotificationHeader*>(stale_mapping);
                    const bool same_segment =
                        stale_header->magic == kNvlinkForwardNotificationMagic &&
                        stale_header->version == kNvlinkForwardNotificationVersion &&
                        stale_header->node_rank == config_.node_rank &&
                        stale_header->gpu_index == config_.local_gpu_index;
                    const bool live_owner = same_segment && process_is_alive(atomic_load_i32(&stale_header->pid));
                    munmap(stale_mapping, nvlink_forward_notification_size_);
                    if (live_owner) {
                        close(stale_fd);
                        throw std::runtime_error("NVLink notification shared memory already has a live owner: " +
                                                 nvlink_forward_notification_name_);
                    }
                }
            }
            close(stale_fd);
        }
        shm_unlink(nvlink_forward_notification_name_.c_str());
        fd = shm_open(nvlink_forward_notification_name_.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    }
    if (fd < 0) {
        throw std::runtime_error("failed to create NVLink notification shared memory " +
                                 nvlink_forward_notification_name_ +
                                 ": errno=" + std::to_string(errno));
    }
    if (ftruncate(fd, static_cast<off_t>(nvlink_forward_notification_size_)) != 0) {
        const auto error = errno;
        close(fd);
        shm_unlink(nvlink_forward_notification_name_.c_str());
        throw std::runtime_error("failed to size NVLink notification shared memory " +
                                 nvlink_forward_notification_name_ +
                                 ": errno=" + std::to_string(error));
    }

    void* mapping = mmap(
        nullptr,
        nvlink_forward_notification_size_,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        0);
    if (mapping == MAP_FAILED) {
        const auto error = errno;
        close(fd);
        shm_unlink(nvlink_forward_notification_name_.c_str());
        throw std::runtime_error("failed to map NVLink notification shared memory " +
                                 nvlink_forward_notification_name_ +
                                 ": errno=" + std::to_string(error));
    }

    std::memset(mapping, 0, nvlink_forward_notification_size_);
    auto* header = reinterpret_cast<NvlinkForwardNotificationHeader*>(mapping);
    header->magic = kNvlinkForwardNotificationMagic;
    header->version = kNvlinkForwardNotificationVersion;
    header->node_rank = config_.node_rank;
    header->gpu_index = config_.local_gpu_index;
    header->num_gpus_per_node = static_cast<uint32_t>(config_.num_gpus_per_node);
    header->queue_count = static_cast<uint32_t>(queue_count);
    header->queue_capacity = static_cast<uint32_t>(queue_capacity);
    header->header_bytes = static_cast<uint32_t>(sizeof(NvlinkForwardNotificationHeader));
    header->queue_bytes = static_cast<uint32_t>(sizeof(NvlinkForwardNotificationQueue));
    header->notification_bytes = static_cast<uint32_t>(sizeof(NvlinkForwardNotification));
    header->entry_bytes = static_cast<uint32_t>(entry_bytes);
    header->reserved = 0;

    auto* queues = reinterpret_cast<NvlinkForwardNotificationQueue*>(
        reinterpret_cast<char*>(mapping) + sizeof(NvlinkForwardNotificationHeader));
    std::size_t queue_index = 0;
    for (int source_gpu = 0; source_gpu < config_.num_gpus_per_node; ++source_gpu) {
        if (source_gpu == config_.local_gpu_index) continue;
        auto* queue = reinterpret_cast<NvlinkForwardNotificationQueue*>(
            reinterpret_cast<char*>(queues) + queue_index * sizeof(NvlinkForwardNotificationQueue));
        queue->source_gpu = source_gpu;
        queue->capacity = static_cast<uint32_t>(queue_capacity);
        atomic_store_u64(&queue->head, 0);
        atomic_store_u64(&queue->tail, 0);
        atomic_store_u64(&queue->dropped, 0);
        atomic_store_u64(&queue->sender_done, 0);
        ++queue_index;
    }
    atomic_store_i32(&header->pid, current_process_id());
    atomic_store_u32(&header->initialized, 1);

    nvlink_forward_notification_fd_ = fd;
    nvlink_forward_notification_header_ = header;
    nvlink_forward_notification_receiver_stop_.store(false);
    nvlink_forward_notifications_sent_.store(0);
    nvlink_forward_notifications_received_.store(0);
    nvlink_forward_notifications_dropped_.store(0);

    RDMA_PROXY_LOG_INFO("created NVLink completion notification shared memory local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " name=", nvlink_forward_notification_name_,
                        " queues=", queue_count,
                        " queue_depth=", queue_capacity,
                        " bytes=", nvlink_forward_notification_size_);
#else
    throw std::runtime_error("NVLink completion notifications require POSIX shared memory");
#endif
}

void Proxy::release_nvlink_forward_notifications() {
#if defined(__unix__) || defined(__APPLE__)
    for (auto& dst : forwarding_notification_destinations_) {
        if (dst.header) {
            munmap(dst.header, dst.bytes);
            dst.header = nullptr;
        }
        if (dst.fd >= 0) {
            close(dst.fd);
            dst.fd = -1;
        }
    }
    forwarding_notification_destinations_.clear();

    if (nvlink_forward_notification_header_) {
        munmap(nvlink_forward_notification_header_, nvlink_forward_notification_size_);
        nvlink_forward_notification_header_ = nullptr;
    }
    if (nvlink_forward_notification_fd_ >= 0) {
        close(nvlink_forward_notification_fd_);
        nvlink_forward_notification_fd_ = -1;
    }
    if (!nvlink_forward_notification_name_.empty()) {
        shm_unlink(nvlink_forward_notification_name_.c_str());
    }
    nvlink_forward_notification_size_ = 0;
    nvlink_forward_notification_name_.clear();
#else
    forwarding_notification_destinations_.clear();
    nvlink_forward_notification_header_ = nullptr;
    nvlink_forward_notification_size_ = 0;
    nvlink_forward_notification_fd_ = -1;
    nvlink_forward_notification_name_.clear();
#endif
}

void Proxy::prepare_forwarding_notification_destinations() {
    if (!config_.nvlink_forward_completion_notifications_enabled) return;

#if defined(__unix__) || defined(__APPLE__)
    forwarding_notification_destinations_.clear();
    forwarding_notification_destinations_.reserve(forwarding_destinations_.size());
    const auto queue_count = static_cast<std::size_t>(config_.num_gpus_per_node - 1);
    const auto entry_bytes = sizeof(NvlinkForwardNotification);
    if (entry_bytes > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("NVLink notification entry size exceeds uint32 range");
    }
    const auto queue_header_bytes = checked_multiply(
        queue_count, sizeof(NvlinkForwardNotificationQueue), "NVLink notification queue headers");
    const auto queue_entry_bytes = checked_multiply(
        checked_multiply(
            queue_count,
            config_.nvlink_forward_notification_queue_depth,
            "NVLink notification queue slots"),
        entry_bytes,
        "NVLink notification queue entry storage");
    const auto expected_size = checked_add(
        checked_add(
            sizeof(NvlinkForwardNotificationHeader),
            queue_header_bytes,
            "NVLink notification shared memory header size"),
        queue_entry_bytes,
        "NVLink notification shared memory size");
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);

    for (const auto& dst : forwarding_destinations_) {
        const auto name = nvlink_forward_notification_shm_name(dst.gpu_index);
        std::string last_wait_reason = "shared memory not present";
        while (std::chrono::steady_clock::now() < deadline) {
            int fd = shm_open(name.c_str(), O_RDWR, 0600);
            if (fd < 0) {
                last_wait_reason = "shm_open errno=" + std::to_string(errno);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            struct stat st {};
            if (fstat(fd, &st) != 0 ||
                st.st_size < static_cast<off_t>(expected_size)) {
                last_wait_reason = "shared memory is not fully sized";
                close(fd);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            void* mapping = mmap(nullptr, expected_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (mapping == MAP_FAILED) {
                last_wait_reason = "mmap errno=" + std::to_string(errno);
                close(fd);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            auto* header = reinterpret_cast<NvlinkForwardNotificationHeader*>(mapping);
            if (atomic_load_u32(&header->initialized) != 1 ||
                header->magic != kNvlinkForwardNotificationMagic ||
                header->version != kNvlinkForwardNotificationVersion ||
                header->node_rank != config_.node_rank ||
                header->gpu_index != dst.gpu_index ||
                header->num_gpus_per_node != static_cast<uint32_t>(config_.num_gpus_per_node) ||
                header->queue_count != static_cast<uint32_t>(queue_count) ||
                header->queue_capacity != static_cast<uint32_t>(config_.nvlink_forward_notification_queue_depth) ||
                header->header_bytes != sizeof(NvlinkForwardNotificationHeader) ||
                header->queue_bytes != sizeof(NvlinkForwardNotificationQueue) ||
                header->notification_bytes != sizeof(NvlinkForwardNotification) ||
                header->entry_bytes != entry_bytes ||
                header->reserved != 0) {
                last_wait_reason = "metadata incomplete or mismatched";
                munmap(mapping, expected_size);
                close(fd);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            const auto receiver_pid = atomic_load_i32(&header->pid);
            if (!process_is_alive(receiver_pid)) {
                last_wait_reason = "receiver PID is not alive";
                munmap(mapping, expected_size);
                close(fd);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            forwarding_notification_destinations_.push_back(
                ForwardNotificationDestinationState{dst.gpu_index, header, expected_size, fd});
            RDMA_PROXY_LOG_INFO("mapped NVLink completion notification queue local_rank=", config_.node_rank,
                                " src_gpu=", config_.local_gpu_index,
                                " dst_gpu=", dst.gpu_index,
                                " receiver_pid=", receiver_pid,
                                " name=", name);
            break;
        }
        if (forwarding_notification_destinations_.empty() ||
            forwarding_notification_destinations_.back().gpu_index != dst.gpu_index) {
            throw std::runtime_error("timed out waiting for NVLink completion notification shared memory " +
                                     name + " reason=" + last_wait_reason);
        }
    }
#else
    throw std::runtime_error("NVLink completion notifications require POSIX shared memory");
#endif
}

void Proxy::initialize_nvlink_forward_notification_dispatch() {
    if (!config_.nvlink_forward_completion_notifications_enabled) return;
    nvlink_forward_notification_dispatch_.reset(new NvlinkForwardNotificationDispatchState);
    nvlink_forward_notification_dispatch_stop_.store(false);
    nvlink_forward_notifications_enqueued_.store(0);
}

void Proxy::enqueue_forward_completion_notifications(
    std::vector<NvlinkForwardNotification>&& notifications) {
    if (!config_.nvlink_forward_completion_notifications_enabled || notifications.empty()) return;
    if (!nvlink_forward_notification_dispatch_) {
        throw std::runtime_error("NVLink completion notification dispatch queue is not initialized");
    }
    {
        std::lock_guard<std::mutex> lock(nvlink_forward_notification_dispatch_->mutex);
        for (auto& notification : notifications) {
            nvlink_forward_notification_dispatch_->pending.push_back(std::move(notification));
        }
    }
    nvlink_forward_notifications_enqueued_.fetch_add(notifications.size());
    if (config_.nvlink_forward_log_batches) {
        RDMA_PROXY_LOG_INFO("nvlink_forward_notifications_enqueued local_rank=", config_.node_rank,
                            " src_gpu=", config_.local_gpu_index,
                            " count=", notifications.size());
    }
}

void Proxy::publish_forward_completion_notification(
    const NvlinkForwardNotification& notification) {
    if (!config_.nvlink_forward_completion_notifications_enabled) return;
    const auto dst_it = std::find_if(
        forwarding_notification_destinations_.begin(),
        forwarding_notification_destinations_.end(),
        [&](const ForwardNotificationDestinationState& dst) {
            return dst.gpu_index == notification.destination_gpu;
        });
    if (dst_it == forwarding_notification_destinations_.end()) {
        throw std::runtime_error("missing NVLink completion notification destination for GPU " +
                                 std::to_string(notification.destination_gpu));
    }

    auto* queue = nvlink_forward_notification_queue_for_source(dst_it->header, config_.local_gpu_index);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    while (true) {
        const auto head = atomic_load_u64(&queue->head);
        const auto tail = atomic_load_u64(&queue->tail);
        if (head - tail < queue->capacity) {
            auto entry = notification;
            entry.sequence = head + 1;
            auto* wire_entry = nvlink_forward_notification_entry(dst_it->header, queue, head);
            *wire_entry = entry;
            atomic_thread_fence_release();
            atomic_store_u64(&queue->head, head + 1);
            nvlink_forward_notifications_sent_.fetch_add(1);
            if (config_.nvlink_forward_log_batches) {
                RDMA_PROXY_LOG_INFO("nvlink_forward_notification_sent iteration=", entry.iteration,
                                    " local_rank=", config_.node_rank,
                                    " src_gpu=", entry.source_gpu,
                                    " dst_gpu=", entry.destination_gpu,
                                    " peer_rank=", entry.peer_rank,
                                    " batch=", entry.batch_index,
                                    " start_token=", entry.start_token,
                                    " tokens=", entry.num_tokens,
                                    " byte_offset=", entry.byte_offset,
                                    " bytes=", entry.bytes,
                                    " sequence=", entry.sequence);
            }
            return;
        }
        if (!process_is_alive(atomic_load_i32(&dst_it->header->pid))) {
            atomic_fetch_add_u64(&queue->dropped, 1);
            nvlink_forward_notifications_dropped_.fetch_add(1);
            throw std::runtime_error("NVLink completion notification receiver is not alive for GPU " +
                                     std::to_string(notification.destination_gpu));
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            atomic_fetch_add_u64(&queue->dropped, 1);
            nvlink_forward_notifications_dropped_.fetch_add(1);
            throw std::runtime_error("timed out waiting for NVLink completion notification queue space dst_gpu=" +
                                     std::to_string(notification.destination_gpu) +
                                     " source_gpu=" + std::to_string(config_.local_gpu_index));
        }
        cpu_relax();
    }
}

void Proxy::mark_forward_notification_senders_done() {
    if (!config_.nvlink_forward_completion_notifications_enabled) return;
    for (const auto& dst : forwarding_notification_destinations_) {
        auto* queue = nvlink_forward_notification_queue_for_source(dst.header, config_.local_gpu_index);
        atomic_store_u64(&queue->sender_done, 1);
        if (config_.nvlink_forward_log_batches) {
            RDMA_PROXY_LOG_INFO("nvlink_forward_notification_sender_done local_rank=", config_.node_rank,
                                " src_gpu=", config_.local_gpu_index,
                                " dst_gpu=", dst.gpu_index);
        }
    }
}

bool Proxy::nvlink_forward_notification_queues_complete() const {
    if (!nvlink_forward_notification_header_) return true;
    auto* base = reinterpret_cast<char*>(nvlink_forward_notification_header_);
    for (std::size_t i = 0; i < nvlink_forward_notification_header_->queue_count; ++i) {
        auto* queue = reinterpret_cast<NvlinkForwardNotificationQueue*>(
            base + nvlink_forward_notification_header_->header_bytes +
            i * nvlink_forward_notification_header_->queue_bytes);
        if (atomic_load_u64(&queue->sender_done) == 0) {
            return false;
        }
        if (atomic_load_u64(&queue->head) != atomic_load_u64(&queue->tail)) {
            return false;
        }
    }
    return true;
}

void Proxy::nvlink_forward_notification_dispatch_loop() {
    try {
        if (!nvlink_forward_notification_dispatch_) return;
        while (true) {
            NvlinkForwardNotification notification;
            bool have_notification = false;
            {
                std::lock_guard<std::mutex> lock(nvlink_forward_notification_dispatch_->mutex);
                if (!nvlink_forward_notification_dispatch_->pending.empty()) {
                    notification = std::move(nvlink_forward_notification_dispatch_->pending.front());
                    nvlink_forward_notification_dispatch_->pending.pop_front();
                    have_notification = true;
                }
            }

            if (have_notification) {
                publish_forward_completion_notification(notification);
                continue;
            }
            if (nvlink_forward_notification_dispatch_stop_.load()) {
                mark_forward_notification_senders_done();
                return;
            }
            // std::this_thread::sleep_for(std::chrono::microseconds(10));
            cpu_relax();
        }
    } catch (const std::exception& e) {
        set_forwarding_error(e.what());
    }
}

std::string Proxy::format_nvlink_forward_notification_log(
    const NvlinkForwardNotification& notification,
    uint64_t dequeue_timestamp_ns) const {
    std::ostringstream out;
    out << "nvlink_forward_notification"
        << " dequeue_timestamp_ns=" << dequeue_timestamp_ns
        << " local_rank=" << config_.node_rank
        << " recv_gpu=" << config_.local_gpu_index
        << " src_gpu=" << notification.source_gpu
        << " dst_gpu=" << notification.destination_gpu
        << " peer_rank=" << notification.peer_rank
        << " peer_slot=" << notification.peer_slot
        << " iteration=" << notification.iteration
        << " batch=" << notification.batch_index
        << " start_token=" << notification.start_token
        << " num_tokens=" << notification.num_tokens
        << " byte_offset=" << notification.byte_offset
        << " bytes=" << notification.bytes
        << " sequence=" << notification.sequence
        << " flags=" << notification.flags;
    return out.str();
}

void Proxy::enqueue_nvlink_forward_notification_log(
    const NvlinkForwardNotification& notification,
    uint64_t dequeue_timestamp_ns) {
    if (!config_.nvlink_forward_notification_log_enabled) return;
    std::lock_guard<std::mutex> lock(nvlink_forward_notification_log_mutex_);
    nvlink_forward_notification_log_queue_.push_back(
        format_nvlink_forward_notification_log(notification, dequeue_timestamp_ns));
}

void Proxy::flush_nvlink_forward_notification_log_queue() {
    if (!config_.nvlink_forward_notification_log_enabled) return;

    std::deque<std::string> messages;
    {
        std::lock_guard<std::mutex> lock(nvlink_forward_notification_log_mutex_);
        messages.swap(nvlink_forward_notification_log_queue_);
    }

    std::filesystem::path path(config_.nvlink_forward_notification_log_dir);
    path /= "nvlink_forward_notifications_rank_" + std::to_string(config_.node_rank) +
        "_gpu_" + std::to_string(config_.local_gpu_index) + ".log";
    if (messages.empty() && std::filesystem::exists(path)) {
        return;
    }

    std::filesystem::create_directories(config_.nvlink_forward_notification_log_dir);
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open NVLink notification log file: " + path.string());
    }
    while (!messages.empty()) {
        out << messages.front() << '\n';
        messages.pop_front();
    }
    out.close();
    if (!out) {
        throw std::runtime_error("failed to flush NVLink notification log file: " + path.string());
    }
    RDMA_PROXY_LOG_INFO("wrote NVLink notification log local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " entries=", nvlink_forward_notifications_received_.load(),
                        " path=", path.string());
}

void Proxy::drain_nvlink_forward_notification_queue(NvlinkForwardNotificationQueue* queue) {
    if (!queue || !nvlink_forward_notification_header_) return;
    while (true) {
        const auto tail = atomic_load_u64(&queue->tail);
        const auto head = atomic_load_u64(&queue->head);
        if (tail >= head) return;
        atomic_thread_fence_acquire();
        auto* wire_entry = nvlink_forward_notification_entry(
            nvlink_forward_notification_header_, queue, tail);
        const auto notification = *wire_entry;
        atomic_store_u64(&queue->tail, tail + 1);
        const auto dequeue_timestamp_ns = unix_epoch_nanoseconds_now();
        nvlink_forward_notifications_received_.fetch_add(1);
        enqueue_nvlink_forward_notification_log(notification, dequeue_timestamp_ns);
        if (config_.nvlink_forward_log_batches) {
            RDMA_PROXY_LOG_INFO("nvlink_forward_notification_received iteration=", notification.iteration,
                                " dequeue_timestamp_ns=", dequeue_timestamp_ns,
                                " local_rank=", config_.node_rank,
                                " recv_gpu=", config_.local_gpu_index,
                                " src_gpu=", notification.source_gpu,
                                " peer_rank=", notification.peer_rank,
                                " batch=", notification.batch_index,
                                " start_token=", notification.start_token,
                                " tokens=", notification.num_tokens,
                                " byte_offset=", notification.byte_offset,
                                " bytes=", notification.bytes,
                                " sequence=", notification.sequence);
        }
    }
}

void Proxy::nvlink_forward_notification_loop() {
    try {
        if (!nvlink_forward_notification_header_) return;
        bool stopping = false;
        auto stop_deadline = std::chrono::steady_clock::time_point{};
        while (true) {
            bool progressed = false;
            auto* base = reinterpret_cast<char*>(nvlink_forward_notification_header_);
            for (std::size_t i = 0; i < nvlink_forward_notification_header_->queue_count; ++i) {
                auto* queue = reinterpret_cast<NvlinkForwardNotificationQueue*>(
                    base + nvlink_forward_notification_header_->header_bytes +
                    i * nvlink_forward_notification_header_->queue_bytes);
                const auto before = atomic_load_u64(&queue->tail);
                drain_nvlink_forward_notification_queue(queue);
                progressed = progressed || atomic_load_u64(&queue->tail) != before;
            }
            if (!progressed) {
                // std::this_thread::sleep_for(std::chrono::microseconds(10));
                cpu_relax();
            }
            if (nvlink_forward_notification_receiver_stop_.load()) {
                if (!stopping) {
                    stopping = true;
                    stop_deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(config_.completion_timeout_ms);
                }
                if (nvlink_forward_notification_queues_complete()) {
                    return;
                }
                if (std::chrono::steady_clock::now() >= stop_deadline) {
                    throw std::runtime_error(
                        "timed out waiting for inbound NVLink notification queues to drain");
                }
            }
        }
    } catch (const std::exception& e) {
        set_forwarding_error(e.what());
    }
}

void Proxy::run_iteration(uint64_t iteration) {
    std::vector<std::vector<ChunkDescriptor>> chunks_by_peer;
    chunks_by_peer.reserve(peers_.size());
    std::size_t total_bytes = 0;
    for (const auto& peer : peers_) {
        chunks_by_peer.push_back(make_chunks(peer.peer_rank));
        for (const auto& chunk : chunks_by_peer.back()) total_bytes += chunk.length_bytes;
    }

    fill_iteration_send_buffers(iteration);

    std::vector<std::vector<QPCompletionBaseline>> baselines;
    baselines.reserve(peers_.size());
    for (std::size_t peer_index = 0; peer_index < peers_.size(); ++peer_index) {
        baselines.push_back(capture_baselines(
            peers_[peer_index], peers_[peer_index].receive_chunks));
    }
    synchronize_iteration_start(iteration);

    const auto start = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<DynamicChunkDistributor>> dispatchers(peers_.size());
    if (config_.sequential_peer_transfers) {
        const auto order = sequential_peer_order();
        for (const auto peer_index : order) {
            dispatchers[peer_index] = enqueue_chunks(
                peers_[peer_index], cuda_buffers_.peer_buffers()[peer_index], chunks_by_peer[peer_index]);
            wait_for_outgoing_transfer(peers_[peer_index], baselines[peer_index], dispatchers[peer_index]);
        }
        for (std::size_t i = 0; i < peers_.size(); ++i) {
            wait_for_iteration(peers_[i], baselines[i], dispatchers[i]);
        }
    } else {
        for (std::size_t i = 0; i < peers_.size(); ++i) {
            dispatchers[i] = enqueue_chunks(
                peers_[i], cuda_buffers_.peer_buffers()[i], chunks_by_peer[i]);
        }
        for (std::size_t i = 0; i < peers_.size(); ++i) {
            wait_for_iteration(peers_[i], baselines[i], dispatchers[i]);
        }
    }

    wait_for_forwarding_iteration(iteration);
    const auto end = std::chrono::steady_clock::now();
    const auto seconds = std::chrono::duration<double>(end - start).count();

    std::vector<IterationAssignment> assignments;
    assignments.reserve(dispatchers.size());
    for (const auto& dispatcher : dispatchers) {
        assignments.push_back(dispatcher->assignment());
    }

    std::size_t verification_errors = 0;
    for (std::size_t i = 0; i < peers_.size(); ++i) {
        verification_errors += verify_immediates(
            peers_[i], peers_[i].receive_chunks, baselines[i], assignments[i], iteration);
    }
    const auto validation_errors = validate_received_data(iteration);

    report_iteration(
        iteration, start, seconds, total_bytes, baselines, assignments,
        verification_errors, validation_errors);
    if (verification_errors != 0) {
        RDMA_PROXY_LOG_WARN("iteration=", iteration,
                            " observed ", verification_errors,
                            " missing/extra data immediate completion count(s); payload validation decides pass/fail");
    }
    if (validation_errors != 0) {
        throw std::runtime_error("iteration " + std::to_string(iteration) +
                                 " completed with " + std::to_string(validation_errors) +
                                 " payload validation errors");
    }
}

void Proxy::fill_iteration_send_buffers(uint64_t iteration) {
    if (!config_.fill_test_data) return;
    if (config_.router_routing_enabled) {
        cuda_buffers_.fill_router_test_pattern(config_.node_rank, iteration);
        return;
    }
    for (const auto& buffers : cuda_buffers_.peer_buffers()) {
        cuda_buffers_.fill_test_pattern(buffers.peer_rank, config_.node_rank, buffers.peer_rank, iteration);
    }
}

void Proxy::synchronize_iteration_start(uint64_t iteration) const {
    if (!peers_.empty()) {
        std::ostringstream local;
        local << "iteration_start"
              << " rank=" << config_.node_rank
              << " gpu=" << config_.local_gpu_index
              << " iteration=" << iteration;
        const auto expected_phase = "iteration_start";
        const auto expected_iteration = "iteration=" + std::to_string(iteration);

        RDMA_PROXY_LOG_INFO("iteration=", iteration, " waiting for ", peers_.size(), " peer start barrier(s)");
        for (const auto& peer : peers_) {
            const auto it = std::find_if(config_.peers.begin(), config_.peers.end(), [&](const PeerAddress& p) {
                return p.node_rank == peer.peer_rank;
            });
            if (it == config_.peers.end()) {
                throw std::runtime_error("cannot synchronize unknown peer rank " + std::to_string(peer.peer_rank));
            }

            const auto remote = connection_manager_.exchange_control_message(
                *it, local.str(), config_.completion_timeout_ms);
            if (remote.find(expected_phase) == std::string::npos ||
                remote.find(expected_iteration) == std::string::npos) {
                throw std::runtime_error("peer start synchronization mismatch: " + remote);
            }
            RDMA_PROXY_LOG_DEBUG("iteration=", iteration, " start synchronized with peer=", peer.peer_rank);
        }
        RDMA_PROXY_LOG_INFO("iteration=", iteration, " peer start barrier complete");
    }
    synchronize_local_iteration_phase("iteration_start", iteration);
}

void Proxy::synchronize_iteration(uint64_t iteration) const {
    if (!peers_.empty()) {
        std::ostringstream local;
        local << "iteration_done"
              << " rank=" << config_.node_rank
              << " gpu=" << config_.local_gpu_index
              << " iteration=" << iteration;
        const auto expected_iteration = "iteration=" + std::to_string(iteration);

        RDMA_PROXY_LOG_INFO("iteration=", iteration, " waiting for ", peers_.size(), " peer synchronization barrier(s)");
        for (const auto& peer : peers_) {
            const auto it = std::find_if(config_.peers.begin(), config_.peers.end(), [&](const PeerAddress& p) {
                return p.node_rank == peer.peer_rank;
            });
            if (it == config_.peers.end()) {
                throw std::runtime_error("cannot synchronize unknown peer rank " + std::to_string(peer.peer_rank));
            }

            const auto remote = connection_manager_.exchange_control_message(
                *it, local.str(), config_.completion_timeout_ms);
            if (remote.find(expected_iteration) == std::string::npos) {
                throw std::runtime_error("peer synchronization iteration mismatch: " + remote);
            }
            RDMA_PROXY_LOG_DEBUG("iteration=", iteration, " synchronized with peer=", peer.peer_rank);
        }
        RDMA_PROXY_LOG_INFO("iteration=", iteration, " peer synchronization complete");
    }
    synchronize_local_iteration_phase("iteration_done", iteration);
}

void Proxy::synchronize_local_iteration_phase(const std::string& phase, uint64_t iteration) const {
    if (!config_.local_iteration_sync_enabled || config_.num_gpus_per_node <= 1) return;
    if (!local_iteration_sync_header_) {
        throw std::runtime_error("local iteration shared memory is not initialized");
    }

    const uint64_t marker = iteration + 1;
    uint64_t Proxy::LocalIterationSyncSlot::* marker_field = nullptr;
    if (phase == "iteration_start") {
        marker_field = &LocalIterationSyncSlot::iteration_start;
    } else if (phase == "iteration_done") {
        marker_field = &LocalIterationSyncSlot::iteration_done;
    } else {
        throw std::runtime_error("unknown local iteration synchronization phase: " + phase);
    }

    auto* local_slot = local_iteration_sync_slot(config_.local_gpu_index);
    atomic_store_i32(&local_slot->gpu_index, config_.local_gpu_index);
    atomic_store_i32(&local_slot->pid, current_process_id());
    atomic_store_u64(&(local_slot->*marker_field), marker);

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    std::vector<std::string> waiting_reasons(static_cast<std::size_t>(config_.num_gpus_per_node));
    RDMA_PROXY_LOG_DEBUG("iteration=", iteration,
                         " waiting for local shared-memory ", phase,
                         " barrier local_rank=", config_.node_rank,
                         " local_gpu=", config_.local_gpu_index,
                         " gpus=", config_.num_gpus_per_node);

    while (true) {
        bool complete = true;
        for (int gpu = 0; gpu < config_.num_gpus_per_node; ++gpu) {
            const auto* slot = local_iteration_sync_slot(gpu);
            const auto pid = atomic_load_i32(&slot->pid);
            const auto slot_gpu = atomic_load_i32(&slot->gpu_index);
            const auto slot_marker = atomic_load_u64(&(slot->*marker_field));
            auto& reason = waiting_reasons[static_cast<std::size_t>(gpu)];
            if (pid <= 0) {
                reason = "missing";
                complete = false;
            } else if (slot_gpu != gpu) {
                reason = "metadata mismatch";
                complete = false;
            } else if (slot_marker != marker) {
                reason = process_is_alive(pid) ?
                    "waiting marker=" + std::to_string(slot_marker) :
                    "process not alive";
                complete = false;
            } else {
                reason.clear();
            }
        }
        if (complete) {
            RDMA_PROXY_LOG_DEBUG("iteration=", iteration,
                                 " local shared-memory ", phase,
                                 " barrier complete local_rank=", config_.node_rank,
                                 " local_gpu=", config_.local_gpu_index);
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            std::ostringstream out;
            out << "timed out waiting for local shared-memory " << phase
                << " barrier iteration=" << iteration
                << " local_rank=" << config_.node_rank
                << " local_gpu=" << config_.local_gpu_index
                << " shm_name=" << local_iteration_sync_name_;
            for (int gpu = 0; gpu < config_.num_gpus_per_node; ++gpu) {
                const auto& reason = waiting_reasons[static_cast<std::size_t>(gpu)];
                if (!reason.empty()) out << " gpu" << gpu << "=" << reason;
            }
            throw std::runtime_error(out.str());
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

std::size_t Proxy::synchronize_local_nvlink_forward_batch_start(
    uint64_t iteration,
    std::size_t batch_index_in_iteration,
    uint64_t batch_sequence,
    std::size_t available_batch_chunks) const {
    if (!config_.nvlink_forward_local_batch_sync_enabled || config_.num_gpus_per_node <= 1) {
        return available_batch_chunks;
    }
    if (!local_iteration_sync_header_) {
        throw std::runtime_error("local shared memory is not initialized for NVLink batch synchronization");
    }

    auto* local_slot = local_iteration_sync_slot(config_.local_gpu_index);
    atomic_store_i32(&local_slot->gpu_index, config_.local_gpu_index);
    atomic_store_i32(&local_slot->pid, current_process_id());
    atomic_store_u64(&local_slot->nvlink_forward_batch_chunks,
                     static_cast<uint64_t>(available_batch_chunks));
    atomic_store_u64(&local_slot->nvlink_forward_batch_start, batch_sequence);

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    RDMA_PROXY_LOG_DEBUG("iteration=", iteration,
                         " waiting for local NVLink forwarding batch-start barrier"
                         " local_rank=", config_.node_rank,
                         " local_gpu=", config_.local_gpu_index,
                         " batch=", batch_index_in_iteration,
                         " sequence=", batch_sequence,
                         " chunks=", available_batch_chunks,
                         " gpus=", config_.num_gpus_per_node);

    uint64_t poll_count = 0;
    std::size_t selected_batch_chunks = 0;
    while (true) {
        bool complete = true;
        std::size_t min_batch_chunks = std::numeric_limits<std::size_t>::max();
        for (int gpu = 0; gpu < config_.num_gpus_per_node; ++gpu) {
            const auto* slot = local_iteration_sync_slot(gpu);
            const auto pid = atomic_load_i32(&slot->pid);
            const auto slot_gpu = atomic_load_i32(&slot->gpu_index);
            const auto slot_marker = atomic_load_u64(&slot->nvlink_forward_batch_start);
            if (pid <= 0 || slot_gpu != gpu || slot_marker < batch_sequence) {
                complete = false;
                break;
            }
            const auto slot_chunks = static_cast<std::size_t>(
                atomic_load_u64(&slot->nvlink_forward_batch_chunks));
            min_batch_chunks = std::min(min_batch_chunks, slot_chunks);
        }
        if (complete) {
            selected_batch_chunks = min_batch_chunks;
            atomic_store_u64(&local_slot->nvlink_forward_batch_selected_chunks,
                             static_cast<uint64_t>(selected_batch_chunks));
            atomic_store_u64(&local_slot->nvlink_forward_batch_selected, batch_sequence);
            break;
        }

        ++poll_count;
        if ((poll_count & 0x3ffULL) == 0 && std::chrono::steady_clock::now() >= deadline) {
            std::ostringstream out;
            out << "timed out waiting for local NVLink forwarding batch-start barrier"
                << " iteration=" << iteration
                << " batch=" << batch_index_in_iteration
                << " sequence=" << batch_sequence
                << " local_rank=" << config_.node_rank
                << " local_gpu=" << config_.local_gpu_index
                << " shm_name=" << local_iteration_sync_name_;
            for (int gpu = 0; gpu < config_.num_gpus_per_node; ++gpu) {
                const auto* slot = local_iteration_sync_slot(gpu);
                const auto pid = atomic_load_i32(&slot->pid);
                const auto slot_gpu = atomic_load_i32(&slot->gpu_index);
                const auto slot_marker = atomic_load_u64(&slot->nvlink_forward_batch_start);
                if (pid <= 0) {
                    out << " gpu" << gpu << "=missing";
                } else if (slot_gpu != gpu) {
                    out << " gpu" << gpu << "=metadata mismatch";
                } else if (slot_marker < batch_sequence) {
                    out << " gpu" << gpu << "="
                        << (process_is_alive(pid) ? "waiting marker=" : "process not alive marker=")
                        << slot_marker;
                }
            }
            throw std::runtime_error(out.str());
        }
        cpu_relax();
    }

    poll_count = 0;
    while (true) {
        bool complete = true;
        for (int gpu = 0; gpu < config_.num_gpus_per_node; ++gpu) {
            const auto* slot = local_iteration_sync_slot(gpu);
            const auto pid = atomic_load_i32(&slot->pid);
            const auto slot_gpu = atomic_load_i32(&slot->gpu_index);
            const auto slot_marker = atomic_load_u64(&slot->nvlink_forward_batch_selected);
            if (pid <= 0 || slot_gpu != gpu || slot_marker < batch_sequence) {
                complete = false;
                break;
            }
        }
        if (complete) {
            RDMA_PROXY_LOG_DEBUG("iteration=", iteration,
                                 " local NVLink forwarding batch-start barrier complete"
                                 " local_rank=", config_.node_rank,
                                 " local_gpu=", config_.local_gpu_index,
                                 " batch=", batch_index_in_iteration,
                                 " sequence=", batch_sequence,
                                 " selected_chunks=", selected_batch_chunks);
            return selected_batch_chunks;
        }

        ++poll_count;
        if ((poll_count & 0x3ffULL) == 0 && std::chrono::steady_clock::now() >= deadline) {
            std::ostringstream out;
            out << "timed out waiting for local NVLink forwarding batch-selected barrier"
                << " iteration=" << iteration
                << " batch=" << batch_index_in_iteration
                << " sequence=" << batch_sequence
                << " local_rank=" << config_.node_rank
                << " local_gpu=" << config_.local_gpu_index
                << " shm_name=" << local_iteration_sync_name_;
            for (int gpu = 0; gpu < config_.num_gpus_per_node; ++gpu) {
                const auto* slot = local_iteration_sync_slot(gpu);
                const auto pid = atomic_load_i32(&slot->pid);
                const auto slot_gpu = atomic_load_i32(&slot->gpu_index);
                const auto slot_marker = atomic_load_u64(&slot->nvlink_forward_batch_selected);
                if (pid <= 0) {
                    out << " gpu" << gpu << "=missing";
                } else if (slot_gpu != gpu) {
                    out << " gpu" << gpu << "=metadata mismatch";
                } else if (slot_marker < batch_sequence) {
                    out << " gpu" << gpu << "="
                        << (process_is_alive(pid) ? "waiting marker=" : "process not alive marker=")
                        << slot_marker;
                }
            }
            throw std::runtime_error(out.str());
        }
        cpu_relax();
    }
}

std::vector<ChunkDescriptor> Proxy::make_chunks(int peer_rank) const {
    if (config_.router_routing_enabled) {
        if (peer_rank < 0) {
            throw std::runtime_error("router chunk construction requires a peer node rank");
        }
        return compute_chunks_from_token_indices(
            router_routing_.token_indices_for_node(peer_rank),
            config_.token_dimension,
            dtype_size(config_.dtype),
            config_.tokens_per_chunk,
            config_.num_qps_per_peer);
    }
    return compute_chunks(
        config_.num_tokens,
        config_.token_dimension,
        dtype_size(config_.dtype),
        config_.tokens_per_chunk,
        config_.num_qps_per_peer,
        effective_rdma_discontinuous_token_payload_enabled(config_));
}

std::vector<std::size_t> Proxy::sequential_peer_order() const {
    std::vector<std::size_t> order;
    order.reserve(peers_.size());
    for (int offset = 1; offset < config_.num_nodes; ++offset) {
        const int target_rank = (config_.node_rank + offset) % config_.num_nodes;
        const auto it = std::find_if(peers_.begin(), peers_.end(), [&](const PeerState& peer) {
            return peer.peer_rank == target_rank;
        });
        if (it == peers_.end()) {
            throw std::runtime_error(
                "sequential peer transfer order cannot find peer rank " + std::to_string(target_rank));
        }
        order.push_back(static_cast<std::size_t>(std::distance(peers_.begin(), it)));
    }
    return order;
}

std::vector<Proxy::QPCompletionBaseline> Proxy::capture_baselines(
    const PeerState& peer,
    const std::vector<ChunkDescriptor>& chunks) const {
    std::vector<QPCompletionBaseline> baselines(peer.workers.size());
    for (std::size_t q = 0; q < peer.workers.size(); ++q) {
        const auto& worker = peer.workers[q];
        auto& baseline = baselines[q];
        baseline.sends = worker->send_completions();
        baseline.recvs = worker->recv_completions();
        baseline.send_markers = worker->send_marker_completions();
        baseline.recv_markers = worker->recv_marker_completions();
        baseline.post_errors = worker->post_errors();
        baseline.cq_errors = worker->cq_errors();
        baseline.unexpected_imms = worker->unexpected_immediate_completions();
        baseline.immediate_counts.resize(chunks.size());
        for (const auto& chunk : chunks) {
            baseline.immediate_counts[chunk.chunk_index] = worker->received_immediate_count(chunk.chunk_index);
        }
    }
    return baselines;
}

std::shared_ptr<DynamicChunkDistributor> Proxy::enqueue_chunks(
    PeerState& peer,
    const PeerGpuBuffers& buffers,
    const std::vector<ChunkDescriptor>& chunks) {
    auto distributor = std::make_shared<DynamicChunkDistributor>(
        chunks,
        peer.peer_rank,
        peer.workers.size(),
        reinterpret_cast<uintptr_t>(buffers.send.ptr),
        peer.local_send_mr.lkey,
        static_cast<uintptr_t>(peer.remote_recv_mr.addr),
        peer.remote_recv_mr.rkey);

    for (std::size_t q = 0; q < peer.workers.size(); ++q) {
        SendTask task;
        task.distributor = distributor;
        peer.workers[q]->enqueue(task);
    }
    RDMA_PROXY_LOG_INFO("started dynamic distributor for ", chunks.size(),
                        " chunks peer=", peer.peer_rank,
                        " qps=", peer.workers.size());
    return distributor;
}

void Proxy::wait_for_iteration(
    const PeerState& peer,
    const std::vector<QPCompletionBaseline>& baselines,
    const std::shared_ptr<DynamicChunkDistributor>& distributor) const {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    while (true) {
        bool complete = true;
        for (std::size_t q = 0; q < peer.workers.size(); ++q) {
            const auto& worker = peer.workers[q];
            if (worker->post_errors() != baselines[q].post_errors ||
                worker->cq_errors() != baselines[q].cq_errors ||
                worker->unexpected_immediate_completions() != baselines[q].unexpected_imms) {
                const auto error = worker->last_error();
                throw std::runtime_error("QP error local_rank=" + std::to_string(config_.node_rank) +
                                         " local_gpu=" + std::to_string(config_.local_gpu_index) +
                                         " remote_rank=" + std::to_string(peer.peer_rank) +
                                         " remote_gpu=" + std::to_string(peer.remote_gpu_index) +
                                         " qp=" + std::to_string(q) +
                                         (error.empty() ? "" : " last_error=" + error));
            }
            if (worker->send_marker_completions() < baselines[q].send_markers + 1 ||
                worker->recv_marker_completions() < baselines[q].recv_markers + 1) {
                complete = false;
            }
        }
        if (complete) return;
        if (std::chrono::steady_clock::now() >= deadline) {
            std::ostringstream out;
            out << "timed out waiting for completions local_rank=" << config_.node_rank
                << " local_gpu=" << config_.local_gpu_index
                << " remote_rank=" << peer.peer_rank
                << " remote_gpu=" << peer.remote_gpu_index;
            const auto assignment = distributor->assignment();
            for (std::size_t q = 0; q < peer.workers.size(); ++q) {
                out << " qp" << q
                    << " send=" << (peer.workers[q]->send_completions() - baselines[q].sends)
                    << "/" << assignment.expected_send_completions_by_qp[q]
                    << " recv=" << (peer.workers[q]->recv_completions() - baselines[q].recvs)
                    << " assigned_chunks=" << assignment.chunks_by_qp[q]
                    << " send_marker=" << (peer.workers[q]->send_marker_completions() - baselines[q].send_markers)
                    << "/1"
                    << " recv_marker=" << (peer.workers[q]->recv_marker_completions() - baselines[q].recv_markers)
                    << "/1";
            }
            out << " assigned_total=" << assignment.assigned_chunks()
                << "/" << distributor->chunk_count();
            throw std::runtime_error(out.str());
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void Proxy::wait_for_outgoing_transfer(
    const PeerState& peer,
    const std::vector<QPCompletionBaseline>& baselines,
    const std::shared_ptr<DynamicChunkDistributor>& distributor) const {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    while (true) {
        bool complete = true;
        for (std::size_t q = 0; q < peer.workers.size(); ++q) {
            const auto& worker = peer.workers[q];
            if (worker->post_errors() != baselines[q].post_errors ||
                worker->cq_errors() != baselines[q].cq_errors ||
                worker->unexpected_immediate_completions() != baselines[q].unexpected_imms) {
                const auto error = worker->last_error();
                throw std::runtime_error("QP error local_rank=" + std::to_string(config_.node_rank) +
                                         " local_gpu=" + std::to_string(config_.local_gpu_index) +
                                         " remote_rank=" + std::to_string(peer.peer_rank) +
                                         " remote_gpu=" + std::to_string(peer.remote_gpu_index) +
                                         (error.empty() ? "" : " last_error=" + error));
            }
            if (worker->send_marker_completions() < baselines[q].send_markers + 1) {
                complete = false;
            }
        }
        if (complete) return;
        if (std::chrono::steady_clock::now() >= deadline) {
            std::ostringstream out;
            out << "timed out waiting for outgoing transfer local_rank=" << config_.node_rank
                << " local_gpu=" << config_.local_gpu_index
                << " remote_rank=" << peer.peer_rank
                << " remote_gpu=" << peer.remote_gpu_index;
            const auto assignment = distributor->assignment();
            for (std::size_t q = 0; q < peer.workers.size(); ++q) {
                out << " qp" << q
                    << " send=" << (peer.workers[q]->send_completions() - baselines[q].sends)
                    << "/" << assignment.expected_send_completions_by_qp[q]
                    << " assigned_chunks=" << assignment.chunks_by_qp[q]
                    << " send_marker=" << (peer.workers[q]->send_marker_completions() - baselines[q].send_markers)
                    << "/1";
            }
            out << " assigned_total=" << assignment.assigned_chunks()
                << "/" << distributor->chunk_count();
            throw std::runtime_error(out.str());
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void Proxy::start_forwarding_thread() {
    if (!config_.nvlink_forwarding_enabled) return;
    if (forwarding_thread_.joinable()) return;

    forwarding_stream_ = create_cuda_stream(
        config_.cuda_device_id,
        config_.nvlink_forward_stream_nonblocking,
        config_.mock_mode);
    initialize_nvlink_forward_notifications();
    publish_local_nvlink_receive_buffers();
    prepare_forwarding_destinations();
    prepare_forwarding_notification_destinations();
    initialize_nvlink_forward_notification_dispatch();
    if (!config_.nvlink_forward_use_round_robin) {
        prepare_forwarding_routing_tables();
    }
    for (const auto& dst : forwarding_destinations_) {
        enable_cuda_peer_access(config_.cuda_device_id, dst.cuda_device_id, config_.mock_mode);
    }

    {
        std::lock_guard<std::mutex> lock(forwarding_mutex_);
        forwarding_next_batch_by_peer_.assign(peers_.size(), 0);
        forwarding_next_chunk_by_peer_.assign(peers_.size(), 0);
        forwarding_compaction_iteration_by_peer_.assign(
            peers_.size(), std::numeric_limits<uint64_t>::max());
        forwarding_compaction_next_source_token_by_peer_.assign(peers_.size(), 0);
        forwarding_compaction_next_destination_token_by_peer_.assign(
            peers_.size(), std::vector<std::size_t>(forwarding_destinations_.size(), 0));
        forwarding_out_of_order_next_batch_by_peer_.assign(peers_.size(), 0);
        forwarding_iteration_stats_.clear();
        forwarding_error_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(nvlink_forward_notification_log_mutex_);
        nvlink_forward_notification_log_queue_.clear();
    }
    forwarding_ready_peer_count_ = peers_.size();
    forwarding_ready_batches_by_peer_.reset(
        new std::atomic<std::size_t>[forwarding_ready_peer_count_]);
    for (std::size_t i = 0; i < forwarding_ready_peer_count_; ++i) {
        forwarding_ready_batches_by_peer_[i].store(0);
    }
    forwarding_ready_chunks_by_peer_.reset(
        new std::atomic<std::size_t>[forwarding_ready_peer_count_]);
    for (std::size_t i = 0; i < forwarding_ready_peer_count_; ++i) {
        forwarding_ready_chunks_by_peer_[i].store(0);
    }
    const bool out_of_order_chunks = config_.nvlink_forward_out_of_order_chunks_enabled;
    forwarding_out_of_order_current_start_by_peer_.reset(
        new std::atomic<std::size_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_current_end_by_peer_.reset(
        new std::atomic<std::size_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_current_length_by_peer_.reset(
        new std::atomic<std::size_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_current_tail_ready_by_peer_.reset(
        new std::atomic<std::size_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_current_version_by_peer_.reset(
        new std::atomic<uint64_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_ready_next_batch_by_peer_.reset(
        new std::atomic<uint64_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_command_start_by_peer_.reset(
        new std::atomic<std::size_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_command_length_by_peer_.reset(
        new std::atomic<std::size_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_command_sequence_by_peer_.reset(
        new std::atomic<uint64_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_command_ack_by_peer_.reset(
        new std::atomic<uint64_t>[forwarding_ready_peer_count_]);
    forwarding_out_of_order_issued_chunks_by_peer_.reset(
        new std::atomic<std::size_t>[forwarding_ready_peer_count_]);
    for (std::size_t i = 0; i < forwarding_ready_peer_count_; ++i) {
        forwarding_out_of_order_current_start_by_peer_[i].store(0);
        forwarding_out_of_order_current_end_by_peer_[i].store(0);
        forwarding_out_of_order_current_length_by_peer_[i].store(0);
        forwarding_out_of_order_current_tail_ready_by_peer_[i].store(0);
        forwarding_out_of_order_current_version_by_peer_[i].store(0);
        forwarding_out_of_order_ready_next_batch_by_peer_[i].store(0);
        forwarding_out_of_order_command_start_by_peer_[i].store(0);
        forwarding_out_of_order_command_length_by_peer_[i].store(0);
        forwarding_out_of_order_command_sequence_by_peer_[i].store(0);
        forwarding_out_of_order_command_ack_by_peer_[i].store(0);
        forwarding_out_of_order_issued_chunks_by_peer_[i].store(0);
    }
    forwarding_out_of_order_peer_states_.clear();
    if (out_of_order_chunks) {
        forwarding_out_of_order_peer_states_.resize(peers_.size());
        for (std::size_t peer_index = 0; peer_index < peers_.size(); ++peer_index) {
            auto& state = forwarding_out_of_order_peer_states_[peer_index];
            const auto chunks_per_iteration = peers_[peer_index].receive_chunks.size();
            const auto total_chunks = config_.num_iterations * chunks_per_iteration;
            state.total_chunks = total_chunks;
            state.chunks_per_iteration = chunks_per_iteration;
            state.chunk_status.reset(new std::atomic<int8_t>[total_chunks]);
            state.arrivals_by_chunk.reset(new std::atomic<uint64_t>[chunks_per_iteration]);
            for (std::size_t i = 0; i < total_chunks; ++i) {
                state.chunk_status[i].store(-1);
            }
            for (std::size_t i = 0; i < chunks_per_iteration; ++i) {
                state.arrivals_by_chunk[i].store(0);
            }
            state.applied_batch_sequence = 0;
        }
    }
    forwarding_batch_available_calls_.store(0);
    forwarding_batch_available_total_ns_.store(0);
    forwarding_batches_issued_.store(0);
    forwarding_stop_.store(false);
    if (config_.nvlink_forward_completion_notifications_enabled) {
        nvlink_forward_notification_dispatch_thread_ =
            std::thread(&Proxy::nvlink_forward_notification_dispatch_loop, this);
        nvlink_forward_notification_thread_ =
            std::thread(&Proxy::nvlink_forward_notification_loop, this);
    }
    forwarding_ready_thread_ = std::thread(&Proxy::forwarding_ready_loop, this);
    forwarding_thread_ = std::thread(&Proxy::forwarding_loop, this);
    const auto forward_threshold_tokens = effective_nvlink_forward_threshold_tokens(config_);
    RDMA_PROXY_LOG_INFO("NVLink forwarding thread started local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " threshold_tokens=", forward_threshold_tokens,
                        " threshold_chunks=", config_.nvlink_forward_threshold_chunks,
                        " min_threshold_chunks=", config_.nvlink_forward_min_threshold_chunks,
                        " max_threshold_chunks=", config_.nvlink_forward_max_threshold_chunks,
                        " out_of_order_chunks=", out_of_order_chunks ? "true" : "false",
                        " chunk_tokens=", config_.nvlink_forward_chunk_tokens,
                        " use_batch_api=", config_.nvlink_forward_use_batch_api ? "true" : "false");
}

void Proxy::stop_forwarding_thread() {
    forwarding_stop_.store(true);
    if (forwarding_ready_thread_.joinable()) forwarding_ready_thread_.join();
    if (forwarding_thread_.joinable()) forwarding_thread_.join();
    nvlink_forward_notification_dispatch_stop_.store(true);
    if (nvlink_forward_notification_dispatch_thread_.joinable()) {
        nvlink_forward_notification_dispatch_thread_.join();
    }
    nvlink_forward_notification_receiver_stop_.store(true);
    if (nvlink_forward_notification_thread_.joinable()) {
        nvlink_forward_notification_thread_.join();
    }
    check_forwarding_error();
    flush_nvlink_forward_notification_log_queue();
    for (auto& dst : forwarding_destinations_) {
        for (auto& source_buffer : dst.source_buffers) {
            if (source_buffer.imported_cuda_ipc) {
                close_cuda_ipc_memory_handle(source_buffer.ptr, config_.mock_mode);
            }
            source_buffer.ptr = nullptr;
            source_buffer.imported_cuda_ipc = false;
        }
    }
    forwarding_destinations_.clear();
    forwarding_ready_batches_by_peer_.reset();
    forwarding_ready_chunks_by_peer_.reset();
    forwarding_out_of_order_current_start_by_peer_.reset();
    forwarding_out_of_order_current_end_by_peer_.reset();
    forwarding_out_of_order_current_length_by_peer_.reset();
    forwarding_out_of_order_current_tail_ready_by_peer_.reset();
    forwarding_out_of_order_current_version_by_peer_.reset();
    forwarding_out_of_order_ready_next_batch_by_peer_.reset();
    forwarding_out_of_order_command_start_by_peer_.reset();
    forwarding_out_of_order_command_length_by_peer_.reset();
    forwarding_out_of_order_command_sequence_by_peer_.reset();
    forwarding_out_of_order_command_ack_by_peer_.reset();
    forwarding_out_of_order_issued_chunks_by_peer_.reset();
    forwarding_ready_peer_count_ = 0;
    forwarding_compaction_iteration_by_peer_.clear();
    forwarding_compaction_next_source_token_by_peer_.clear();
    forwarding_compaction_next_destination_token_by_peer_.clear();
    forwarding_out_of_order_peer_states_.clear();
    nvlink_forward_notification_dispatch_.reset();
    release_nvlink_forward_notifications();
    destroy_cuda_stream(forwarding_stream_, config_.mock_mode);
    forwarding_stream_ = nullptr;
}

void Proxy::prepare_forwarding_routing_tables() {
    std::mt19937_64 rng(config_.nvlink_routing_seed);
    std::bernoulli_distribution route(config_.nvlink_routing_probability);

    for (auto& peer : peers_) {
        if (config_.router_routing_enabled) {
            if (peer.router_x4.size() != forwarding_tokens_for_peer(peer)) {
                throw std::runtime_error("router x4 length does not match received x3 token count");
            }
            peer.forwarding_routing_table = normalize_router_x4_for_nvlink(
                peer.router_x4, config_.num_gpus_per_node);
            RDMA_PROXY_LOG_INFO("installed router x4 NVLink forwarding table peer_rank=", peer.peer_rank,
                                " rows=", peer.forwarding_routing_table.size(),
                                " active_columns=", active_routing_columns(config_.num_gpus_per_node));
            continue;
        }

        std::vector<uint8_t> table(config_.num_tokens, 0);
        for (auto& row : table) {
            for (std::size_t column = 0; column < 8; ++column) {
                if (route(rng)) {
                    row = static_cast<uint8_t>(row | routing_column_mask(column));
                }
            }
        }

        std::sort(table.begin(), table.end(), std::greater<uint8_t>());
        peer.forwarding_routing_table = std::move(table);
        RDMA_PROXY_LOG_INFO("generated NVLink routing table peer_rank=", peer.peer_rank,
                            " rows=", config_.num_tokens,
                            " active_columns=", active_routing_columns(config_.num_gpus_per_node),
                            " ignored_columns=", 8 - active_routing_columns(config_.num_gpus_per_node),
                            " probability=", config_.nvlink_routing_probability,
                            " seed=", config_.nvlink_routing_seed);
    }

}

std::size_t Proxy::forwarding_tokens_for_peer(const PeerState& peer) const {
    if (config_.router_routing_enabled) {
        if (peer.receive_chunks.empty()) return 0;
        const auto& last = peer.receive_chunks.back();
        return last.start_token + last.num_tokens;
    }
    return config_.num_tokens;
}

std::string Proxy::nvlink_exchange_file(int gpu_index) const {
    std::filesystem::path path(config_.nvlink_forward_exchange_dir);
    path /= "node_" + std::to_string(config_.node_rank) + "_gpu_" + std::to_string(gpu_index) + ".txt";
    return path.string();
}

void Proxy::publish_local_nvlink_receive_buffers() const {
    if (!config_.nvlink_forward_destinations.empty()) return;

    std::filesystem::create_directories(config_.nvlink_forward_exchange_dir);
    const auto path = nvlink_exchange_file(config_.local_gpu_index);
    const auto tmp_path = path + ".tmp." + std::to_string(current_process_id());
    std::ofstream out(tmp_path, std::ios::trunc);
    if (!out) throw std::runtime_error("failed to write NVLink exchange file: " + tmp_path);
    out << "node_rank " << config_.node_rank << '\n'
        << "gpu_index " << config_.local_gpu_index << '\n'
        << "exporter_pid " << current_process_id() << '\n'
        << "cuda_device_id " << config_.cuda_device_id << '\n'
        << "buffer_bytes " << cuda_buffers_.nvlink_receive_buffer_bytes() << '\n';
    for (const auto& entry : cuda_buffers_.nvlink_receive_buffers()) {
        if (entry.source_node_rank >= 0) {
            out << "source_node " << entry.source_node_rank << ' ';
        }
        out << "source_gpu " << entry.source_gpu_index
            << " mock_addr " << reinterpret_cast<uintptr_t>(entry.recv.ptr)
            << " ipc_handle " << export_cuda_ipc_memory_handle(entry.recv.ptr, config_.mock_mode)
            << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("failed to flush NVLink exchange file: " + tmp_path);
    std::filesystem::rename(tmp_path, path);
    RDMA_PROXY_LOG_INFO("published NVLink receive buffers local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " path=", path,
                        " buffers=", cuda_buffers_.nvlink_receive_buffers().size(),
                        " bytes_per_buffer=", cuda_buffers_.nvlink_receive_buffer_bytes());
}

Proxy::ForwardDestinationState Proxy::load_forward_destination(int dst_gpu) const {
    if (!config_.nvlink_forward_destinations.empty()) {
        const auto it = std::find_if(
            config_.nvlink_forward_destinations.begin(),
            config_.nvlink_forward_destinations.end(),
            [&](const NvlinkForwardDestination& dst) { return dst.gpu_index == dst_gpu; });
        if (it == config_.nvlink_forward_destinations.end()) {
            throw std::runtime_error("missing manual NVLink forwarding destination for GPU " + std::to_string(dst_gpu));
        }
        ForwardDestinationState destination;
        destination.gpu_index = it->gpu_index;
        destination.cuda_device_id = it->cuda_device_id;
        destination.source_buffers.push_back(ForwardDestinationBufferState{
            -1,
            reinterpret_cast<void*>(static_cast<uintptr_t>(it->buffer_addr)),
            it->buffer_bytes,
            false});
        return destination;
    }

    struct PublishedBuffer {
        int source_node_rank{-1};
        int source_gpu_index{-1};
        uint64_t mock_addr{0};
        std::string ipc_handle;
    };

    const auto path = nvlink_exchange_file(dst_gpu);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    std::string last_wait_reason = "file not present";
    while (std::chrono::steady_clock::now() < deadline) {
        if (!std::filesystem::exists(path)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::ifstream in(path);
        if (!in) {
            last_wait_reason = "file not readable";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        int node_rank = -1;
        int gpu_index = -1;
        int exporter_pid = -1;
        int cuda_device_id = -1;
        std::size_t buffer_bytes = 0;
        std::vector<PublishedBuffer> published_buffers;
        std::string key;
        while (in >> key) {
            if (key == "node_rank") {
                in >> node_rank;
            } else if (key == "gpu_index") {
                in >> gpu_index;
            } else if (key == "exporter_pid") {
                in >> exporter_pid;
            } else if (key == "cuda_device_id") {
                in >> cuda_device_id;
            } else if (key == "buffer_bytes") {
                in >> buffer_bytes;
            } else if (key == "source_node") {
                PublishedBuffer entry;
                std::string source_gpu_key;
                std::string mock_key;
                std::string handle_key;
                in >> entry.source_node_rank
                   >> source_gpu_key >> entry.source_gpu_index
                   >> mock_key >> entry.mock_addr
                   >> handle_key >> entry.ipc_handle;
                if (!in || source_gpu_key != "source_gpu" ||
                    mock_key != "mock_addr" || handle_key != "ipc_handle") {
                    throw std::runtime_error("malformed node-aware NVLink source buffer entry in " + path);
                }
                published_buffers.push_back(std::move(entry));
            } else if (key == "source_gpu") {
                PublishedBuffer entry;
                std::string mock_key;
                std::string handle_key;
                in >> entry.source_gpu_index
                   >> mock_key >> entry.mock_addr
                   >> handle_key >> entry.ipc_handle;
                if (!in || mock_key != "mock_addr" || handle_key != "ipc_handle") {
                    throw std::runtime_error("malformed NVLink source buffer entry in " + path);
                }
                published_buffers.push_back(std::move(entry));
            } else {
                throw std::runtime_error("unknown key in NVLink exchange file " + path + ": " + key);
            }
        }

        if (!process_is_alive(exporter_pid)) {
            last_wait_reason = "exporter PID is not alive";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        std::vector<const PublishedBuffer*> selected_buffers;
        if (config_.router_routing_enabled) {
            selected_buffers.reserve(config_.peers.size());
            for (const auto& peer : config_.peers) {
                const auto entry = std::find_if(
                    published_buffers.begin(),
                    published_buffers.end(),
                    [&](const PublishedBuffer& candidate) {
                        return candidate.source_node_rank == peer.node_rank &&
                            candidate.source_gpu_index == config_.local_gpu_index;
                    });
                if (entry != published_buffers.end()) selected_buffers.push_back(&*entry);
            }
        } else {
            const auto entry = std::find_if(
                published_buffers.begin(),
                published_buffers.end(),
                [&](const PublishedBuffer& candidate) {
                    return candidate.source_node_rank < 0 &&
                        candidate.source_gpu_index == config_.local_gpu_index;
                });
            if (entry != published_buffers.end()) selected_buffers.push_back(&*entry);
        }
        const auto expected_buffer_count = config_.router_routing_enabled ?
            config_.peers.size() : std::size_t{1};
        if (node_rank != config_.node_rank || gpu_index != dst_gpu || cuda_device_id < 0 ||
            buffer_bytes == 0 || selected_buffers.size() != expected_buffer_count) {
            last_wait_reason = "metadata incomplete or mismatched";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        ForwardDestinationState destination;
        destination.gpu_index = dst_gpu;
        destination.cuda_device_id = cuda_device_id;
        destination.source_buffers.reserve(selected_buffers.size());
        try {
            for (const auto* published : selected_buffers) {
                void* ptr = open_cuda_ipc_memory_handle(
                    published->ipc_handle, published->mock_addr, config_.mock_mode);
                destination.source_buffers.push_back(ForwardDestinationBufferState{
                    published->source_node_rank,
                    ptr,
                    buffer_bytes,
                    true});
                RDMA_PROXY_LOG_INFO(
                    "imported NVLink receive buffer local_rank=", config_.node_rank,
                    " source_node=", published->source_node_rank,
                    " src_gpu=", config_.local_gpu_index,
                    " dst_gpu=", dst_gpu,
                    " exporter_pid=", exporter_pid,
                    " cuda_device_id=", cuda_device_id,
                    " bytes=", buffer_bytes,
                    " ptr=", reinterpret_cast<uintptr_t>(ptr),
                    " path=", path);
            }
        } catch (...) {
            for (auto& source_buffer : destination.source_buffers) {
                if (source_buffer.imported_cuda_ipc) {
                    close_cuda_ipc_memory_handle(source_buffer.ptr, config_.mock_mode);
                }
            }
            throw;
        }
        return destination;
    }

    throw std::runtime_error("timed out waiting for live NVLink exchange file: " + path +
                             " reason=" + last_wait_reason);
}

const Proxy::ForwardDestinationBufferState& Proxy::forward_destination_buffer(
    const ForwardDestinationState& destination,
    int source_node_rank) const {
    const auto exact = std::find_if(
        destination.source_buffers.begin(),
        destination.source_buffers.end(),
        [&](const ForwardDestinationBufferState& buffer) {
            return buffer.source_node_rank == source_node_rank;
        });
    if (exact != destination.source_buffers.end()) return *exact;

    const auto wildcard = std::find_if(
        destination.source_buffers.begin(),
        destination.source_buffers.end(),
        [](const ForwardDestinationBufferState& buffer) {
            return buffer.source_node_rank < 0;
        });
    if (wildcard != destination.source_buffers.end()) return *wildcard;

    throw std::runtime_error(
        "missing NVLink forwarding destination buffer for source node " +
        std::to_string(source_node_rank) + " and destination GPU " +
        std::to_string(destination.gpu_index));
}

void Proxy::prepare_forwarding_destinations() {
    forwarding_destinations_.clear();
    forwarding_destinations_.reserve(static_cast<std::size_t>(config_.num_gpus_per_node - 1));
    for (std::size_t chunk_index = 0;
         chunk_index < static_cast<std::size_t>(config_.num_gpus_per_node - 1);
         ++chunk_index) {
        const int dst_gpu =
            (config_.local_gpu_index + 1 + static_cast<int>(chunk_index)) % config_.num_gpus_per_node;
        forwarding_destinations_.push_back(load_forward_destination(dst_gpu));
    }
}

std::vector<std::size_t> Proxy::nvlink_forward_peer_order() const {
    std::vector<std::size_t> order;
    order.reserve(peers_.size());
    for (int offset = config_.num_nodes - 1; offset >= 1; --offset) {
        const int target_rank = (config_.node_rank + offset) % config_.num_nodes;
        const auto it = std::find_if(peers_.begin(), peers_.end(), [&](const PeerState& peer) {
            return peer.peer_rank == target_rank;
        });
        if (it == peers_.end()) {
            throw std::runtime_error(
                "NVLink forwarding peer order cannot find peer rank " + std::to_string(target_rank));
        }
        order.push_back(static_cast<std::size_t>(std::distance(peers_.begin(), it)));
    }
    return order;
}

bool Proxy::forwarding_batch_available(
    const PeerState& peer,
    const std::vector<ChunkDescriptor>& chunks,
    std::size_t batch_start_token,
    std::size_t batch_tokens,
    uint64_t required_count) const {
    if (chunks.empty() || batch_tokens == 0) return false;
    const auto batch_end_token = batch_start_token + batch_tokens;
    const auto first_chunk = batch_start_token / config_.tokens_per_chunk;
    const auto last_chunk = (batch_end_token - 1) / config_.tokens_per_chunk;
    if (first_chunk >= chunks.size()) return false;

    bool overlaps_any_chunk = false;
    const auto last_existing_chunk = std::min(last_chunk, chunks.size() - 1);
    for (std::size_t chunk_pos = first_chunk; chunk_pos <= last_existing_chunk; ++chunk_pos) {
        const auto& chunk = chunks[chunk_pos];
        const auto chunk_end_token = chunk.start_token + chunk.num_tokens;
        if (chunk.start_token >= batch_end_token || chunk_end_token <= batch_start_token) {
            continue;
        }
        overlaps_any_chunk = true;

        uint64_t total = 0;
        for (const auto& worker : peer.workers) {
            total += worker->received_immediate_count(chunk.chunk_index);
        }
        if (total < required_count) {
            return false;
        }
    }
    return overlaps_any_chunk;
}

bool Proxy::forwarding_chunk_available(
    const PeerState& peer,
    const ChunkDescriptor& chunk,
    uint64_t required_count) const {
    uint64_t total = 0;
    for (const auto& worker : peer.workers) {
        total += worker->received_immediate_count(chunk.chunk_index);
    }
    return total >= required_count;
}

void Proxy::issue_forwarding_batch(
    const PeerState& peer,
    const PeerGpuBuffers& buffers,
    uint64_t iteration,
    std::size_t batch_index_in_iteration,
    std::size_t batch_start_token,
    std::size_t batch_tokens) {
    const auto token_bytes = config_.token_dimension * dtype_size(config_.dtype);
    const auto batch_timing_start = std::chrono::steady_clock::now();
    std::size_t batch_bytes = 0;
    std::vector<NvlinkForwardNotification> completed_notifications;
    if (batch_start_token + batch_tokens > forwarding_tokens_for_peer(peer)) {
        throw std::runtime_error("NVLink forwarding batch exceeds token range");
    }

    const auto peer_buffer_it = std::find_if(
        cuda_buffers_.peer_buffers().cbegin(),
        cuda_buffers_.peer_buffers().cend(),
        [&](const PeerGpuBuffers& entry) { return &entry == &buffers; });
    if (peer_buffer_it == cuda_buffers_.peer_buffers().cend()) {
        throw std::runtime_error("cannot find NVLink forwarding peer buffer slot");
    }
    const auto peer_slot = static_cast<std::size_t>(
        std::distance(cuda_buffers_.peer_buffers().cbegin(), peer_buffer_it));
    const auto legacy_peer_slot_offset = cuda_buffers_.token_buffer_bytes() * peer_slot;
    const bool compact_router_destinations =
        config_.router_routing_enabled && !config_.nvlink_forward_use_round_robin;
    if (compact_router_destinations) {
        if (peer_slot >= forwarding_compaction_iteration_by_peer_.size() ||
            peer_slot >= forwarding_compaction_next_source_token_by_peer_.size() ||
            peer_slot >= forwarding_compaction_next_destination_token_by_peer_.size() ||
            forwarding_compaction_next_destination_token_by_peer_[peer_slot].size() !=
                forwarding_destinations_.size()) {
            throw std::runtime_error("missing router NVLink destination compaction state");
        }
        auto& cursor_iteration = forwarding_compaction_iteration_by_peer_[peer_slot];
        if (cursor_iteration != iteration) {
            cursor_iteration = iteration;
            forwarding_compaction_next_source_token_by_peer_[peer_slot] = 0;
            std::fill(
                forwarding_compaction_next_destination_token_by_peer_[peer_slot].begin(),
                forwarding_compaction_next_destination_token_by_peer_[peer_slot].end(),
                std::size_t{0});
        }
        if (batch_start_token != forwarding_compaction_next_source_token_by_peer_[peer_slot]) {
            throw std::runtime_error(
                "router NVLink forwarding batch is out of order for peer " +
                std::to_string(peer.peer_rank) + ": expected source token " +
                std::to_string(forwarding_compaction_next_source_token_by_peer_[peer_slot]) +
                ", got " + std::to_string(batch_start_token));
        }
    }

    if (config_.nvlink_forward_use_round_robin) {
        for (std::size_t chunk_index = 0;
             chunk_index < static_cast<std::size_t>(config_.num_gpus_per_node - 1);
             ++chunk_index) {
            const int dst_gpu =
                (config_.local_gpu_index + 1 + static_cast<int>(chunk_index)) % config_.num_gpus_per_node;
            const auto dst_it = std::find_if(
                forwarding_destinations_.begin(),
                forwarding_destinations_.end(),
                [&](const ForwardDestinationState& dst) { return dst.gpu_index == dst_gpu; });
            if (dst_it == forwarding_destinations_.end()) {
                throw std::runtime_error("missing NVLink forwarding destination for GPU " + std::to_string(dst_gpu));
            }
            const auto& destination_buffer = forward_destination_buffer(*dst_it, peer.peer_rank);

            const auto token_offset = batch_start_token + chunk_index * config_.nvlink_forward_chunk_tokens;
            const auto source_byte_offset = token_offset * token_bytes;
            const auto destination_byte_offset = legacy_peer_slot_offset + source_byte_offset;
            const auto bytes = config_.nvlink_forward_chunk_tokens * token_bytes;
            if (source_byte_offset + bytes > buffers.recv.bytes ||
                destination_byte_offset + bytes > destination_buffer.bytes) {
                throw std::runtime_error("NVLink forwarding copy exceeds source or destination buffer size");
            }

            CudaForwardCopy copy;
            copy.src = static_cast<const char*>(buffers.recv.ptr) + source_byte_offset;
            copy.dst = static_cast<char*>(destination_buffer.ptr) + destination_byte_offset;
            copy.bytes = bytes;
            batch_bytes += bytes;

            if (config_.nvlink_forward_log_batches) {
                RDMA_PROXY_LOG_INFO("nvlink_forward_round_robin iteration=", iteration,
                                    " local_rank=", config_.node_rank,
                                    " src_gpu=", config_.local_gpu_index,
                                    " dst_gpu=", dst_gpu,
                                    " peer_rank=", peer.peer_rank,
                                    " batch=", batch_index_in_iteration,
                                    " chunk=", chunk_index,
                                    " token_offset=", token_offset,
                                    " token_count=", config_.nvlink_forward_chunk_tokens,
                                    " peer_slot_offset=", legacy_peer_slot_offset,
                                    " bytes=", bytes,
                                    " src_addr=", reinterpret_cast<uintptr_t>(copy.src),
                                    " dst_addr=", reinterpret_cast<uintptr_t>(copy.dst));
            }
            launch_cuda_forward_copy_batch_async(
                copy,
                forwarding_stream_,
                config_.nvlink_forward_use_batch_api,
                config_.mock_mode);
            if (config_.nvlink_forward_completion_notifications_enabled) {
                NvlinkForwardNotification notification;
                notification.iteration = iteration;
                notification.batch_index = batch_index_in_iteration;
                notification.peer_slot = peer_slot;
                notification.start_token = token_offset;
                notification.num_tokens = config_.nvlink_forward_chunk_tokens;
                notification.byte_offset = destination_byte_offset;
                notification.bytes = bytes;
                notification.source_gpu = config_.local_gpu_index;
                notification.destination_gpu = dst_gpu;
                notification.peer_rank = peer.peer_rank;
                notification.flags = kNvlinkForwardRoundRobinFlag;
                completed_notifications.push_back(notification);
            }
        }
    } else {
        const auto& routing_table = peer.forwarding_routing_table;
        if (routing_table.size() != forwarding_tokens_for_peer(peer)) {
            throw std::runtime_error("missing or mis-sized NVLink routing table for peer");
        }

        // Each routing row is one uint8_t for one received token. Columns
        // 0..num_gpus_per_node-2 map to local destination GPUs via
        // (local_gpu + 1 + column) % num_gpus_per_node. The remaining columns
        // are still generated and included in sorting, but are ignored for
        // forwarding because they do not represent unique peer GPUs.
        for (std::size_t destination_index = 0;
             destination_index < forwarding_destinations_.size();
             ++destination_index) {
            const auto& dst = forwarding_destinations_[destination_index];
            const auto& destination_buffer = forward_destination_buffer(dst, peer.peer_rank);
            // Exact source-node buffers are private to one global source GPU.
            // A manual wildcard destination retains the legacy peer-slot layout.
            const auto destination_slot_offset = destination_buffer.source_node_rank >= 0 ?
                std::size_t{0} : legacy_peer_slot_offset;
            const auto route_column =
                routing_column_for_gpu(config_.local_gpu_index, dst.gpu_index, config_.num_gpus_per_node);
            const int mapped_gpu =
                routing_column_to_gpu(config_.local_gpu_index, route_column, config_.num_gpus_per_node);
            if (mapped_gpu != dst.gpu_index) {
                throw std::runtime_error("NVLink routing destination mapping mismatch");
            }

            std::vector<CudaForwardCopy> copies;
            copies.reserve(batch_tokens);
            const auto destination_start_token = compact_router_destinations ?
                forwarding_compaction_next_destination_token_by_peer_[peer_slot][destination_index] :
                batch_start_token;
            const auto destination_base_offset =
                destination_slot_offset + destination_start_token * token_bytes;
            std::size_t routed_tokens = 0;
            std::size_t run_start_token = 0;
            std::size_t run_tokens = 0;
            auto flush_run = [&]() {
                if (run_tokens == 0) return;
                const auto source_byte_offset = run_start_token * token_bytes;
                const auto destination_byte_offset = destination_base_offset + routed_tokens * token_bytes;
                const auto bytes = run_tokens * token_bytes;
                if (source_byte_offset + bytes > buffers.recv.bytes ||
                    destination_byte_offset + bytes > destination_buffer.bytes) {
                    throw std::runtime_error("NVLink forwarding copy exceeds source or destination buffer size");
                }

                CudaForwardCopy copy;
                copy.src = static_cast<const char*>(buffers.recv.ptr) + source_byte_offset;
                copy.dst = static_cast<char*>(destination_buffer.ptr) + destination_byte_offset;
                copy.bytes = bytes;
                copies.push_back(copy);
                routed_tokens += run_tokens;
                run_tokens = 0;
            };

            for (std::size_t token = 0; token < batch_tokens; ++token) {
                const auto token_index = batch_start_token + token;
                if ((routing_table[token_index] & routing_column_mask(route_column)) == 0) {
                    flush_run();
                    continue;
                }

                if (run_tokens == 0) {
                    run_start_token = token_index;
                }
                ++run_tokens;
            }
            flush_run();

            if (compact_router_destinations) {
                forwarding_compaction_next_destination_token_by_peer_[peer_slot][destination_index] +=
                    routed_tokens;
            }

            if (copies.empty()) {
                if (config_.nvlink_forward_log_batches) {
                    RDMA_PROXY_LOG_INFO("nvlink_forward_route_empty iteration=", iteration,
                                        " local_rank=", config_.node_rank,
                                        " src_gpu=", config_.local_gpu_index,
                                        " dst_gpu=", dst.gpu_index,
                                        " peer_rank=", peer.peer_rank,
                                        " batch=", batch_index_in_iteration,
                                        " route_column=", route_column);
                }
                continue;
            }

            const auto bytes = routed_tokens * token_bytes;
            batch_bytes += bytes;
            if (config_.nvlink_forward_log_batches) {
                RDMA_PROXY_LOG_INFO("nvlink_forward iteration=", iteration,
                                    " local_rank=", config_.node_rank,
                                    " src_gpu=", config_.local_gpu_index,
                                    " dst_gpu=", dst.gpu_index,
                                    " peer_rank=", peer.peer_rank,
                                    " batch=", batch_index_in_iteration,
                                    " route_column=", route_column,
                                    " batch_start_token=", batch_start_token,
                                    " destination_start_token=", destination_start_token,
                                    " routed_tokens=", routed_tokens,
                                    " batch_entries=", copies.size(),
                                    " peer_slot_offset=", destination_slot_offset,
                                    " bytes=", bytes,
                                    " first_src_addr=", reinterpret_cast<uintptr_t>(copies.front().src),
                                    " first_dst_addr=", reinterpret_cast<uintptr_t>(copies.front().dst));
            }
            launch_cuda_forward_copy_batch_async(
                copies,
                forwarding_stream_,
                config_.nvlink_forward_use_batch_api,
                config_.mock_mode);
            if (config_.nvlink_forward_completion_notifications_enabled) {
                NvlinkForwardNotification notification;
                notification.iteration = iteration;
                notification.batch_index = batch_index_in_iteration;
                notification.peer_slot = peer_slot;
                notification.start_token = destination_start_token;
                notification.num_tokens = routed_tokens;
                notification.byte_offset = destination_base_offset;
                notification.bytes = bytes;
                notification.source_gpu = config_.local_gpu_index;
                notification.destination_gpu = dst.gpu_index;
                notification.peer_rank = peer.peer_rank;
                notification.flags = 0;
                completed_notifications.push_back(notification);
            }
        }
        if (compact_router_destinations) {
            forwarding_compaction_next_source_token_by_peer_[peer_slot] += batch_tokens;
        }
    }
    if (config_.nvlink_forward_synchronize_batches) {
        synchronize_cuda_stream(forwarding_stream_, config_.mock_mode);
        const auto batch_timing_end = std::chrono::steady_clock::now();
        enqueue_forward_completion_notifications(std::move(completed_notifications));
        const auto batch_seconds = std::chrono::duration<double>(batch_timing_end - batch_timing_start).count();
        const double batch_gbytes_per_sec = batch_seconds > 0.0 ?
            static_cast<double>(batch_bytes) / batch_seconds / 1.0e9 : 0.0;
        const double batch_gbits_per_sec = batch_gbytes_per_sec * 8.0;
        {
            std::lock_guard<std::mutex> lock(forwarding_mutex_);
            if (forwarding_iteration_stats_.size() <= iteration) {
                forwarding_iteration_stats_.resize(static_cast<std::size_t>(iteration) + 1);
            }
            auto& stats = forwarding_iteration_stats_[static_cast<std::size_t>(iteration)];
            ++stats.batch_count;
            stats.total_bytes += batch_bytes;
            if (batch_bytes > 0) {
                ++stats.bandwidth_sample_count;
                stats.total_seconds += batch_seconds;
                stats.sum_batch_bandwidth_gbytes_per_sec += batch_gbytes_per_sec;
                stats.sum_batch_bandwidth_gbits_per_sec += batch_gbits_per_sec;
            }
        }
        if (config_.nvlink_forward_log_batches) {
            RDMA_PROXY_LOG_INFO("nvlink_forward_batch_complete iteration=", iteration,
                                " local_rank=", config_.node_rank,
                                " local_gpu=", config_.local_gpu_index,
                                " peer_rank=", peer.peer_rank,
                                " batch=", batch_index_in_iteration,
                                " bytes=", batch_bytes,
                                " elapsed_us=", static_cast<uint64_t>(batch_seconds * 1.0e6),
                                " bandwidth_GBps=", std::fixed, std::setprecision(3), batch_gbytes_per_sec,
                                " bandwidth_gbps=", std::fixed, std::setprecision(3), batch_gbits_per_sec);
        }
    }
}

void Proxy::forwarding_ready_loop() {
    try {
        const auto peer_order = nvlink_forward_peer_order();
        const bool dynamic_threshold = nvlink_forward_dynamic_threshold_enabled(config_);
        const auto forward_threshold_tokens = effective_nvlink_forward_threshold_tokens(config_);
        const bool finite_iterations = config_.num_iterations != 0;

        while (!forwarding_stop_.load()) {
            check_forwarding_error();
            bool progressed = false;
            for (const auto peer_index : peer_order) {
                if (peer_index >= forwarding_ready_peer_count_) {
                    throw std::runtime_error("NVLink forwarding ready peer index out of range");
                }

                const auto& peer = peers_[peer_index];
                const auto& chunks = peer.receive_chunks;
                const auto chunks_per_iteration = chunks.size();
                const auto forwarding_tokens = forwarding_tokens_for_peer(peer);
                const auto batches_per_iteration = dynamic_threshold || forwarding_tokens == 0 ?
                    std::size_t{0} :
                    (forwarding_tokens + forward_threshold_tokens - 1) / forward_threshold_tokens;
                const auto total_batches = finite_iterations ?
                    config_.num_iterations * batches_per_iteration : std::size_t{0};
                const auto total_chunks = finite_iterations ?
                    config_.num_iterations * chunks_per_iteration : std::size_t{0};
                if (chunks.empty()) continue;

                if (dynamic_threshold && config_.nvlink_forward_out_of_order_chunks_enabled) {
                    if (peer_index >= forwarding_out_of_order_peer_states_.size()) {
                        throw std::runtime_error("missing NVLink out-of-order forwarding state");
                    }
                    auto& state = forwarding_out_of_order_peer_states_[peer_index];
                    uint64_t pending_ack_sequence = 0;

                    const auto command_sequence =
                        forwarding_out_of_order_command_sequence_by_peer_[peer_index].load();
                    if (command_sequence > state.applied_batch_sequence) {
                        const auto command_start =
                            forwarding_out_of_order_command_start_by_peer_[peer_index].load();
                        const auto command_length =
                            forwarding_out_of_order_command_length_by_peer_[peer_index].load();
                        if (command_length == 0 || command_start + command_length > state.total_chunks) {
                            throw std::runtime_error("invalid NVLink out-of-order forwarded range command");
                        }
                        for (std::size_t i = 0; i < command_length; ++i) {
                            state.chunk_status[command_start + i].store(1);
                        }
                        state.applied_batch_sequence = command_sequence;
                        pending_ack_sequence = command_sequence;
                    }

                    if (pending_ack_sequence == 0) {
                        const auto published_length =
                            forwarding_out_of_order_current_length_by_peer_[peer_index].load();
                        const bool published_tail_ready =
                            forwarding_out_of_order_current_tail_ready_by_peer_[peer_index].load() != 0;
                        if (published_length >= config_.nvlink_forward_min_threshold_chunks ||
                            (published_tail_ready && published_length > 0)) {
                            continue;
                        }
                    }

                    const auto issued_chunks =
                        forwarding_out_of_order_issued_chunks_by_peer_[peer_index].load();
                    if (finite_iterations && issued_chunks >= total_chunks) {
                        continue;
                    }
                    const auto active_iteration = issued_chunks / chunks_per_iteration;
                    const auto active_iteration_start = active_iteration * chunks_per_iteration;
                    const auto active_iteration_end = active_iteration_start + chunks_per_iteration;

                    std::size_t best_start = 0;
                    std::size_t best_length = 0;
                    bool best_tail_ready = false;
                    bool any_unforwarded = false;
                    bool all_unforwarded_ready = true;
                    for (std::size_t global_chunk = active_iteration_start;
                         global_chunk < active_iteration_end;
                         ++global_chunk) {
                        const auto status = state.chunk_status[global_chunk].load();
                        if (status == 1) continue;
                        any_unforwarded = true;
                        if (status == -1) {
                            all_unforwarded_ready = false;
                        }
                    }
                    std::size_t run_start = 0;
                    std::size_t run_length = 0;
                    auto flush_run = [&]() {
                        if (run_length == 0) return;
                        if (best_length == 0 &&
                            (run_length >= config_.nvlink_forward_min_threshold_chunks ||
                             (any_unforwarded && all_unforwarded_ready))) {
                            best_start = run_start;
                            best_length = run_length;
                            best_tail_ready = any_unforwarded && all_unforwarded_ready;
                        }
                        run_length = 0;
                    };
                    for (std::size_t global_chunk = active_iteration_start;
                         global_chunk < active_iteration_end;
                         ++global_chunk) {
                        const bool available = state.chunk_status[global_chunk].load() == 0;
                        if (!available) {
                            flush_run();
                            if (best_length != 0) break;
                            continue;
                        }
                        if (run_length == 0) {
                            run_start = global_chunk;
                        }
                        ++run_length;
                    }
                    if (best_length == 0) {
                        flush_run();
                    }

                    const auto current_version =
                        forwarding_out_of_order_current_version_by_peer_[peer_index].load();
                    forwarding_out_of_order_current_version_by_peer_[peer_index].store(current_version + 1);
                    forwarding_out_of_order_current_start_by_peer_[peer_index].store(best_start);
                    forwarding_out_of_order_current_end_by_peer_[peer_index].store(
                        best_length == 0 ? best_start : best_start + best_length - 1);
                    forwarding_out_of_order_current_length_by_peer_[peer_index].store(best_length);
                    forwarding_out_of_order_current_tail_ready_by_peer_[peer_index].store(
                        best_tail_ready ? 1 : 0);
                    forwarding_out_of_order_current_version_by_peer_[peer_index].store(current_version + 2);
                    if (pending_ack_sequence != 0) {
                        forwarding_out_of_order_ready_next_batch_by_peer_[peer_index].store(pending_ack_sequence);
                        forwarding_out_of_order_command_ack_by_peer_[peer_index].store(pending_ack_sequence);
                    }
                    progressed = true;
                } else if (dynamic_threshold) {
                    auto next_ready_chunk =
                        forwarding_ready_chunks_by_peer_[peer_index].load();
                    if (finite_iterations && next_ready_chunk >= total_chunks) {
                        continue;
                    }

                    std::size_t advanced = 0;
                    const auto availability_start = std::chrono::steady_clock::now();
                    while (!forwarding_stop_.load()) {
                        if (finite_iterations && next_ready_chunk >= total_chunks) {
                            break;
                        }
                        const uint64_t iteration =
                            static_cast<uint64_t>(next_ready_chunk / chunks_per_iteration);
                        const auto chunk_in_iteration = next_ready_chunk % chunks_per_iteration;
                        const auto required_count = iteration + 1;
                        if (!forwarding_chunk_available(
                                peer,
                                chunks[chunk_in_iteration],
                                required_count)) {
                            break;
                        }
                        ++next_ready_chunk;
                        ++advanced;
                    }
                    const auto availability_elapsed_ns =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - availability_start).count();
                    forwarding_batch_available_calls_.fetch_add(1);
                    forwarding_batch_available_total_ns_.fetch_add(
                        static_cast<uint64_t>(std::max<int64_t>(0, availability_elapsed_ns)));

                    if (advanced == 0) {
                        continue;
                    }

                    forwarding_ready_chunks_by_peer_[peer_index].store(next_ready_chunk);
                    progressed = true;
                } else {
                    const auto next_batch =
                        forwarding_ready_batches_by_peer_[peer_index].load();
                    if (finite_iterations && next_batch >= total_batches) {
                        continue;
                    }

                    const uint64_t iteration = static_cast<uint64_t>(next_batch / batches_per_iteration);
                    const auto batch_in_iteration = next_batch % batches_per_iteration;
                    const auto batch_start_token =
                        batch_in_iteration * forward_threshold_tokens;
                    const auto batch_tokens =
                        std::min(forward_threshold_tokens, forwarding_tokens - batch_start_token);
                    const auto required_count = iteration + 1;

                    const auto availability_start = std::chrono::steady_clock::now();
                    const bool batch_available = forwarding_batch_available(
                        peer,
                        chunks,
                        batch_start_token,
                        batch_tokens,
                        required_count);
                    const auto availability_elapsed_ns =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - availability_start).count();
                    forwarding_batch_available_calls_.fetch_add(1);
                    forwarding_batch_available_total_ns_.fetch_add(
                        static_cast<uint64_t>(std::max<int64_t>(0, availability_elapsed_ns)));

                    if (!batch_available) {
                        continue;
                    }

                    forwarding_ready_batches_by_peer_[peer_index].store(next_batch + 1);
                    progressed = true;
                }
            }
            if (!progressed) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    } catch (const std::exception& e) {
        set_forwarding_error(e.what());
    }
}

void Proxy::record_out_of_order_forwarding_arrival(
    std::size_t peer_index,
    std::size_t chunk_index) {
    if (!config_.nvlink_forward_out_of_order_chunks_enabled ||
        forwarding_stop_.load() ||
        peer_index >= forwarding_out_of_order_peer_states_.size()) {
        return;
    }
    auto& state = forwarding_out_of_order_peer_states_[peer_index];
    if (!state.arrivals_by_chunk || !state.chunk_status ||
        chunk_index >= state.chunks_per_iteration) {
        return;
    }
    const auto arrival_count = state.arrivals_by_chunk[chunk_index].fetch_add(1) + 1;
    if (arrival_count == 0 || arrival_count > config_.num_iterations) {
        return;
    }
    const auto global_chunk =
        static_cast<std::size_t>(arrival_count - 1) * state.chunks_per_iteration + chunk_index;
    if (global_chunk >= state.total_chunks) {
        return;
    }
    int8_t expected = -1;
    state.chunk_status[global_chunk].compare_exchange_strong(expected, 0);
}

void Proxy::forwarding_loop() {
    try {
        const auto peer_order = nvlink_forward_peer_order();
        const bool dynamic_threshold = nvlink_forward_dynamic_threshold_enabled(config_);
        const auto forward_threshold_tokens = effective_nvlink_forward_threshold_tokens(config_);
        const bool finite_iterations = config_.num_iterations != 0;
        uint64_t local_batch_sync_sequence = 0;

        while (!forwarding_stop_.load()) {
            check_forwarding_error();
            bool progressed = false;
            for (const auto peer_index : peer_order) {
                if (peer_index >= forwarding_ready_peer_count_) {
                    throw std::runtime_error("NVLink forwarding ready peer index out of range");
                }
                const auto& peer = peers_[peer_index];
                const auto& chunks = peer.receive_chunks;
                const auto chunks_per_iteration = chunks.size();
                const auto forwarding_tokens = forwarding_tokens_for_peer(peer);
                const auto batches_per_iteration = dynamic_threshold || forwarding_tokens == 0 ?
                    std::size_t{0} :
                    (forwarding_tokens + forward_threshold_tokens - 1) / forward_threshold_tokens;
                const auto total_batches = finite_iterations ?
                    config_.num_iterations * batches_per_iteration : std::size_t{0};
                const auto total_chunks = finite_iterations ?
                    config_.num_iterations * chunks_per_iteration : std::size_t{0};
                if (chunks.empty()) continue;

                if (dynamic_threshold && config_.nvlink_forward_out_of_order_chunks_enabled) {
                    uint64_t next_batch = 0;
                    {
                        std::lock_guard<std::mutex> lock(forwarding_mutex_);
                        next_batch = forwarding_out_of_order_next_batch_by_peer_.at(peer_index);
                    }
                    const auto ready_next_batch =
                        forwarding_out_of_order_ready_next_batch_by_peer_[peer_index].load();
                    if (ready_next_batch != next_batch) {
                        continue;
                    }
                    if (finite_iterations) {
                        const auto issued_chunks =
                            forwarding_out_of_order_issued_chunks_by_peer_[peer_index].load();
                        if (issued_chunks >= total_chunks) {
                            continue;
                        }
                    }

                    std::size_t current_start = 0;
                    std::size_t current_length = 0;
                    bool tail_ready = false;
                    bool snapshot_valid = false;
                    for (int attempt = 0; attempt < 4; ++attempt) {
                        const auto version_before =
                            forwarding_out_of_order_current_version_by_peer_[peer_index].load();
                        if ((version_before & 1ULL) != 0) {
                            cpu_relax();
                            continue;
                        }
                        current_start =
                            forwarding_out_of_order_current_start_by_peer_[peer_index].load();
                        current_length =
                            forwarding_out_of_order_current_length_by_peer_[peer_index].load();
                        tail_ready =
                            forwarding_out_of_order_current_tail_ready_by_peer_[peer_index].load() != 0;
                        const auto version_after =
                            forwarding_out_of_order_current_version_by_peer_[peer_index].load();
                        if (version_before == version_after && (version_after & 1ULL) == 0) {
                            snapshot_valid = true;
                            break;
                        }
                        cpu_relax();
                    }
                    if (!snapshot_valid) {
                        continue;
                    }
                    if (current_length == 0) {
                        continue;
                    }

                    std::size_t batch_chunks =
                        std::min(current_length, config_.nvlink_forward_max_threshold_chunks);
                    if (batch_chunks < config_.nvlink_forward_min_threshold_chunks) {
                        if (!tail_ready) {
                            continue;
                        }
                        batch_chunks = current_length;
                    }
                    if (batch_chunks == 0) {
                        continue;
                    }

                    const uint64_t iteration = static_cast<uint64_t>(current_start / chunks_per_iteration);
                    const auto chunk_in_iteration = current_start % chunks_per_iteration;
                    if (chunk_in_iteration + batch_chunks > chunks_per_iteration) {
                        throw std::runtime_error("NVLink out-of-order forwarding batch crosses iteration boundary");
                    }

                    if (config_.nvlink_forward_local_batch_sync_enabled) {
                        ++local_batch_sync_sequence;
                        batch_chunks = synchronize_local_nvlink_forward_batch_start(
                            iteration,
                            chunk_in_iteration,
                            local_batch_sync_sequence,
                            batch_chunks);
                    }
                    if (batch_chunks == 0 || batch_chunks > current_length ||
                        chunk_in_iteration + batch_chunks > chunks_per_iteration) {
                        throw std::runtime_error("local NVLink forwarding batch synchronization selected "
                                                 "an invalid out-of-order batch size");
                    }

                    const auto command_sequence = next_batch + 1;
                    forwarding_out_of_order_command_start_by_peer_[peer_index].store(current_start);
                    forwarding_out_of_order_command_length_by_peer_[peer_index].store(batch_chunks);
                    forwarding_out_of_order_command_sequence_by_peer_[peer_index].store(command_sequence);

                    const auto batch_start_token = chunks[chunk_in_iteration].start_token;
                    const auto last_chunk = chunk_in_iteration + batch_chunks - 1;
                    const auto batch_end_token =
                        chunks[last_chunk].start_token + chunks[last_chunk].num_tokens;
                    const auto batch_tokens = batch_end_token - batch_start_token;

                    if (config_.nvlink_forward_log_batches) {
                        RDMA_PROXY_LOG_INFO("nvlink_forward_out_of_order_batch_ready iteration=", iteration,
                                            " local_rank=", config_.node_rank,
                                            " local_gpu=", config_.local_gpu_index,
                                            " peer_rank=", peer.peer_rank,
                                            " batch_sequence=", command_sequence,
                                            " batch_start_chunk=", chunk_in_iteration,
                                            " global_start_chunk=", current_start,
                                            " batch_chunks=", batch_chunks,
                                            " available_run_chunks=", current_length,
                                            " batch_start_token=", batch_start_token,
                                            " batch_tokens=", batch_tokens);
                    }
                    issue_forwarding_batch(
                        peer,
                        cuda_buffers_.peer_buffers()[peer_index],
                        iteration,
                        chunk_in_iteration,
                        batch_start_token,
                        batch_tokens);
                    {
                        std::lock_guard<std::mutex> lock(forwarding_mutex_);
                        forwarding_out_of_order_next_batch_by_peer_.at(peer_index) = command_sequence;
                    }
                    forwarding_out_of_order_issued_chunks_by_peer_[peer_index].fetch_add(batch_chunks);
                    forwarding_batches_issued_.fetch_add(1);
                    progressed = true;
                } else if (dynamic_threshold) {
                    std::size_t next_chunk = 0;
                    {
                        std::lock_guard<std::mutex> lock(forwarding_mutex_);
                        next_chunk = forwarding_next_chunk_by_peer_.at(peer_index);
                    }
                    if (finite_iterations && next_chunk >= total_chunks) {
                        continue;
                    }

                    const uint64_t iteration = static_cast<uint64_t>(next_chunk / chunks_per_iteration);
                    const auto chunk_in_iteration = next_chunk % chunks_per_iteration;
                    const auto iteration_end_chunk =
                        (static_cast<std::size_t>(iteration) + 1) * chunks_per_iteration;
                    const auto ready_chunks =
                        forwarding_ready_chunks_by_peer_[peer_index].load();
                    if (next_chunk >= ready_chunks) {
                        continue;
                    }

                    const auto available_chunks =
                        std::min(ready_chunks, iteration_end_chunk) - next_chunk;
                    const auto remaining_chunks = iteration_end_chunk - next_chunk;
                    std::size_t batch_chunks =
                        std::min(available_chunks, config_.nvlink_forward_max_threshold_chunks);
                    const bool final_iteration_tail_ready =
                        available_chunks == remaining_chunks &&
                        remaining_chunks < config_.nvlink_forward_min_threshold_chunks;
                    if (batch_chunks < config_.nvlink_forward_min_threshold_chunks) {
                        if (!final_iteration_tail_ready) {
                            continue;
                        }
                        batch_chunks = remaining_chunks;
                    }
                    if (batch_chunks == 0) {
                        continue;
                    }

                    if (config_.nvlink_forward_local_batch_sync_enabled) {
                        ++local_batch_sync_sequence;
                        batch_chunks = synchronize_local_nvlink_forward_batch_start(
                            iteration,
                            chunk_in_iteration,
                            local_batch_sync_sequence,
                            batch_chunks);
                    }
                    if (batch_chunks == 0 || batch_chunks > available_chunks) {
                        throw std::runtime_error("local NVLink forwarding batch synchronization selected "
                                                 "an invalid dynamic batch size");
                    }

                    const auto batch_start_token = chunks[chunk_in_iteration].start_token;
                    const auto last_chunk = chunk_in_iteration + batch_chunks - 1;
                    const auto batch_end_token =
                        chunks[last_chunk].start_token + chunks[last_chunk].num_tokens;
                    const auto batch_tokens = batch_end_token - batch_start_token;

                    if (config_.nvlink_forward_log_batches) {
                        RDMA_PROXY_LOG_INFO("nvlink_forward_dynamic_batch_ready iteration=", iteration,
                                            " local_rank=", config_.node_rank,
                                            " local_gpu=", config_.local_gpu_index,
                                            " peer_rank=", peer.peer_rank,
                                            " batch_start_chunk=", chunk_in_iteration,
                                            " batch_chunks=", batch_chunks,
                                            " available_chunks=", available_chunks,
                                            " batch_start_token=", batch_start_token,
                                            " batch_tokens=", batch_tokens);
                    }
                    issue_forwarding_batch(
                        peer,
                        cuda_buffers_.peer_buffers()[peer_index],
                        iteration,
                        chunk_in_iteration,
                        batch_start_token,
                        batch_tokens);
                    {
                        std::lock_guard<std::mutex> lock(forwarding_mutex_);
                        forwarding_next_chunk_by_peer_.at(peer_index) = next_chunk + batch_chunks;
                    }
                    forwarding_batches_issued_.fetch_add(1);
                    progressed = true;
                } else {
                    std::size_t next_batch = 0;
                    {
                        std::lock_guard<std::mutex> lock(forwarding_mutex_);
                        next_batch = forwarding_next_batch_by_peer_.at(peer_index);
                    }
                    if (finite_iterations && next_batch >= total_batches) {
                        continue;
                    }

                    const uint64_t iteration = static_cast<uint64_t>(next_batch / batches_per_iteration);
                    const auto batch_in_iteration = next_batch % batches_per_iteration;
                    const auto batch_start_token =
                        batch_in_iteration * forward_threshold_tokens;
                    const auto batch_tokens =
                        std::min(forward_threshold_tokens, forwarding_tokens - batch_start_token);
                    const auto ready_batches =
                        forwarding_ready_batches_by_peer_[peer_index].load();
                    if (next_batch >= ready_batches) {
                        continue;
                    }

                    if (config_.nvlink_forward_log_batches) {
                        RDMA_PROXY_LOG_INFO("nvlink_forward_batch_ready iteration=", iteration,
                                            " local_rank=", config_.node_rank,
                                            " local_gpu=", config_.local_gpu_index,
                                            " peer_rank=", peer.peer_rank,
                                            " batch=", batch_in_iteration,
                                            " batch_start_token=", batch_start_token,
                                            " batch_tokens=", batch_tokens);
                    }
                    if (config_.nvlink_forward_local_batch_sync_enabled) {
                        ++local_batch_sync_sequence;
                        synchronize_local_nvlink_forward_batch_start(
                            iteration,
                            batch_in_iteration,
                            local_batch_sync_sequence,
                            0);
                    }
                    issue_forwarding_batch(
                        peer,
                        cuda_buffers_.peer_buffers()[peer_index],
                        iteration,
                        batch_in_iteration,
                        batch_start_token,
                        batch_tokens);
                    {
                        std::lock_guard<std::mutex> lock(forwarding_mutex_);
                        forwarding_next_batch_by_peer_.at(peer_index) = next_batch + 1;
                    }
                    forwarding_batches_issued_.fetch_add(1);
                    progressed = true;
                }
            }
            if (!progressed) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    } catch (const std::exception& e) {
        set_forwarding_error(e.what());
    }
}

void Proxy::wait_for_forwarding_iteration(uint64_t iteration) {
    if (!config_.nvlink_forwarding_enabled) return;
    const bool dynamic_threshold = nvlink_forward_dynamic_threshold_enabled(config_);
    const bool out_of_order_chunks =
        dynamic_threshold && config_.nvlink_forward_out_of_order_chunks_enabled;
    const auto forward_threshold_tokens = effective_nvlink_forward_threshold_tokens(config_);
    const auto batches_per_iteration =
        dynamic_threshold ? std::size_t{0} :
        (config_.num_tokens + forward_threshold_tokens - 1) / forward_threshold_tokens;
    const auto chunks_per_iteration = peers_.empty() ?
        std::size_t{0} : peers_.front().receive_chunks.size();
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    while (true) {
        check_forwarding_error();
        bool complete = true;
        {
            std::lock_guard<std::mutex> lock(forwarding_mutex_);
            if (out_of_order_chunks) {
                for (std::size_t i = 0; i < forwarding_ready_peer_count_; ++i) {
                    const auto required_chunks = static_cast<std::size_t>(iteration + 1) *
                        peers_[i].receive_chunks.size();
                    if (forwarding_out_of_order_issued_chunks_by_peer_[i].load() < required_chunks) {
                        complete = false;
                        break;
                    }
                }
            } else if (dynamic_threshold) {
                for (std::size_t i = 0; i < forwarding_next_chunk_by_peer_.size(); ++i) {
                    const auto required_chunks = static_cast<std::size_t>(iteration + 1) *
                        peers_[i].receive_chunks.size();
                    if (forwarding_next_chunk_by_peer_[i] < required_chunks) {
                        complete = false;
                        break;
                    }
                }
            } else {
                for (std::size_t i = 0; i < forwarding_next_batch_by_peer_.size(); ++i) {
                    const auto forwarding_tokens = forwarding_tokens_for_peer(peers_[i]);
                    const auto peer_batches = forwarding_tokens == 0 ? std::size_t{0} :
                        (forwarding_tokens + forward_threshold_tokens - 1) / forward_threshold_tokens;
                    const auto required_batches =
                        static_cast<std::size_t>(iteration + 1) * peer_batches;
                    if (forwarding_next_batch_by_peer_[i] < required_batches) {
                        complete = false;
                        break;
                    }
                }
            }
        }
        if (complete) {
            if (config_.nvlink_forward_synchronize_iteration) {
                synchronize_cuda_stream(forwarding_stream_, config_.mock_mode);
            }
            ForwardingIterationStats stats;
            if (config_.nvlink_forward_synchronize_batches) {
                std::lock_guard<std::mutex> lock(forwarding_mutex_);
                if (forwarding_iteration_stats_.size() > iteration) {
                    stats = forwarding_iteration_stats_[static_cast<std::size_t>(iteration)];
                }
            }
            const double avg_batch_gbytes_per_sec = stats.bandwidth_sample_count > 0 ?
                stats.sum_batch_bandwidth_gbytes_per_sec /
                    static_cast<double>(stats.bandwidth_sample_count) : 0.0;
            const double avg_batch_gbits_per_sec = stats.bandwidth_sample_count > 0 ?
                stats.sum_batch_bandwidth_gbits_per_sec /
                    static_cast<double>(stats.bandwidth_sample_count) : 0.0;
            const double aggregate_gbytes_per_sec = stats.total_seconds > 0.0 ?
                static_cast<double>(stats.total_bytes) / stats.total_seconds / 1.0e9 : 0.0;
            const double aggregate_gbits_per_sec = aggregate_gbytes_per_sec * 8.0;
            if (config_.nvlink_forward_synchronize_batches) {
                RDMA_PROXY_LOG_INFO("nvlink_forward_iteration_complete iteration=", iteration,
                                    " local_rank=", config_.node_rank,
                                    " local_gpu=", config_.local_gpu_index,
                                    " batches_per_peer=", dynamic_threshold ? std::size_t{0} : batches_per_iteration,
                                    " chunks_per_peer=", dynamic_threshold ? chunks_per_iteration : std::size_t{0},
                                    " dynamic_batch_count=", dynamic_threshold ? stats.batch_count : std::size_t{0},
                                    " out_of_order_chunks=", out_of_order_chunks ? "true" : "false",
                                    " synchronized_batch_count=", stats.batch_count,
                                    " bandwidth_sample_count=", stats.bandwidth_sample_count,
                                    " empty_bandwidth_sample_count=",
                                        stats.batch_count - stats.bandwidth_sample_count,
                                    " synchronized_batch_bytes=", stats.total_bytes,
                                    " average_batch_bandwidth_GBps=", std::fixed, std::setprecision(3),
                                        avg_batch_gbytes_per_sec,
                                    " average_batch_bandwidth_gbps=", std::fixed, std::setprecision(3),
                                        avg_batch_gbits_per_sec,
                                    " aggregate_synchronized_bandwidth_GBps=", std::fixed, std::setprecision(3),
                                        aggregate_gbytes_per_sec,
                                    " aggregate_synchronized_bandwidth_gbps=", std::fixed, std::setprecision(3),
                                        aggregate_gbits_per_sec);
            } else {
                RDMA_PROXY_LOG_INFO("nvlink_forward_iteration_complete iteration=", iteration,
                                    " local_rank=", config_.node_rank,
                                    " local_gpu=", config_.local_gpu_index,
                                    " batches_per_peer=", dynamic_threshold ? std::size_t{0} : batches_per_iteration,
                                    " chunks_per_peer=", dynamic_threshold ? chunks_per_iteration : std::size_t{0},
                                    " out_of_order_chunks=", out_of_order_chunks ? "true" : "false");
            }
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            std::ostringstream out;
            out << "timed out waiting for NVLink forwarding iteration=" << iteration
                << " local_rank=" << config_.node_rank
                << " local_gpu=" << config_.local_gpu_index;
            std::lock_guard<std::mutex> lock(forwarding_mutex_);
            if (out_of_order_chunks) {
                for (std::size_t i = 0; i < forwarding_ready_peer_count_; ++i) {
                    const auto required_chunks = static_cast<std::size_t>(iteration + 1) *
                        peers_[i].receive_chunks.size();
                    out << " peer" << peers_[i].peer_rank
                        << "_issued_chunks=" << forwarding_out_of_order_issued_chunks_by_peer_[i].load()
                        << "/" << required_chunks;
                }
            } else if (dynamic_threshold) {
                for (std::size_t i = 0; i < forwarding_next_chunk_by_peer_.size(); ++i) {
                    const auto required_chunks = static_cast<std::size_t>(iteration + 1) *
                        peers_[i].receive_chunks.size();
                    out << " peer" << peers_[i].peer_rank
                        << "_chunks=" << forwarding_next_chunk_by_peer_[i]
                        << "/" << required_chunks;
                }
            } else {
                for (std::size_t i = 0; i < forwarding_next_batch_by_peer_.size(); ++i) {
                    const auto forwarding_tokens = forwarding_tokens_for_peer(peers_[i]);
                    const auto peer_batches = forwarding_tokens == 0 ? std::size_t{0} :
                        (forwarding_tokens + forward_threshold_tokens - 1) / forward_threshold_tokens;
                    const auto required_batches =
                        static_cast<std::size_t>(iteration + 1) * peer_batches;
                    out << " peer" << peers_[i].peer_rank
                        << "_batches=" << forwarding_next_batch_by_peer_[i]
                        << "/" << required_batches;
                }
            }
            throw std::runtime_error(out.str());
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void Proxy::set_forwarding_error(const std::string& error) {
    std::lock_guard<std::mutex> lock(forwarding_mutex_);
    forwarding_error_ = error;
}

void Proxy::check_forwarding_error() const {
    std::lock_guard<std::mutex> lock(forwarding_mutex_);
    if (!forwarding_error_.empty()) {
        throw std::runtime_error("NVLink forwarding failed: " + forwarding_error_);
    }
}

std::size_t Proxy::verify_immediates(
    const PeerState& peer,
    const std::vector<ChunkDescriptor>& chunks,
    const std::vector<QPCompletionBaseline>& baselines,
    const IterationAssignment& assignment,
    uint64_t iteration) const {
    std::size_t errors = 0;
    for (const auto& chunk : chunks) {
        uint64_t observed_total = 0;
        uint64_t baseline_total = 0;
        for (std::size_t q = 0; q < peer.workers.size(); ++q) {
            observed_total += peer.workers[q]->received_immediate_count(chunk.chunk_index);
            baseline_total += baselines[q].immediate_counts[chunk.chunk_index];
        }
        const auto expected = baseline_total + 1;
        if (observed_total != expected) {
            ++errors;
            RDMA_PROXY_LOG_WARN("iteration=", iteration,
                                " peer=", peer.peer_rank,
                                " chunk=", chunk.chunk_index,
                                " local_send_qp=",
                                chunk.chunk_index < assignment.qp_by_chunk.size() ?
                                    assignment.qp_by_chunk[chunk.chunk_index] : -1,
                                " immediate_count=", observed_total,
                                " expected=", expected);
        }
    }
    return errors;
}

std::size_t Proxy::validate_received_data(uint64_t iteration) const {
    if (!config_.validate_data) return 0;
    std::size_t errors = 0;
    for (std::size_t peer_index = 0; peer_index < peers_.size(); ++peer_index) {
        const auto& peer = peers_[peer_index];
        const auto& buffers = cuda_buffers_.peer_buffers()[peer_index];
        const auto& chunks = peer.receive_chunks;
        std::string error;
        const int expected_source = config_.mock_mode ? config_.node_rank : buffers.peer_rank;
        const int expected_destination = config_.router_routing_enabled ? -1 :
            (config_.mock_mode ? buffers.peer_rank : config_.node_rank);
        const bool valid = effective_rdma_discontinuous_token_payload_enabled(config_) ?
            cuda_buffers_.validate_recv_pattern(
                buffers.peer_rank, expected_source, expected_destination, iteration, chunks, &error) :
            cuda_buffers_.validate_recv_pattern(
                buffers.peer_rank, expected_source, expected_destination, iteration, &error);
        if (!valid) {
            ++errors;
            RDMA_PROXY_LOG_ERROR("receive validation failed: ", error);
        }
    }
    return errors;
}

void Proxy::report_rdma_bandwidth_summary() const {
    if (rdma_iteration_bandwidth_gbps_.empty()) return;

    double sum = 0.0;
    for (const auto bandwidth : rdma_iteration_bandwidth_gbps_) {
        sum += bandwidth;
    }

    const auto count = rdma_iteration_bandwidth_gbps_.size();
    const double average = sum / static_cast<double>(count);
    double variance = 0.0;
    for (const auto bandwidth : rdma_iteration_bandwidth_gbps_) {
        const auto delta = bandwidth - average;
        variance += delta * delta;
    }
    variance /= static_cast<double>(count);

    auto sorted = rdma_iteration_bandwidth_gbps_;
    std::sort(sorted.begin(), sorted.end());
    const double median = count % 2 == 0 ?
        (sorted[count / 2 - 1] + sorted[count / 2]) * 0.5 :
        sorted[count / 2];
    const auto forwarding_availability_calls = forwarding_batch_available_calls_.load();
    const auto forwarding_availability_total_ns = forwarding_batch_available_total_ns_.load();
    const auto forwarding_batches_issued = forwarding_batches_issued_.load();
    const double forwarding_availability_avg_us_per_call =
        forwarding_availability_calls > 0 ?
            static_cast<double>(forwarding_availability_total_ns) /
                static_cast<double>(forwarding_availability_calls) / 1000.0 :
            0.0;
    const double forwarding_availability_avg_us_per_batch =
        forwarding_batches_issued > 0 ?
            static_cast<double>(forwarding_availability_total_ns) /
                static_cast<double>(forwarding_batches_issued) / 1000.0 :
            0.0;

    std::ostringstream summary;
    summary << std::fixed << std::setprecision(3)
            << "rdma_bandwidth_summary"
            << " local_rank=" << config_.node_rank
            << " local_gpu=" << config_.local_gpu_index
            << " iterations=" << count
            << " average_bandwidth_gbps=" << average
            << " min_bandwidth_gbps=" << sorted.front()
            << " max_bandwidth_gbps=" << sorted.back()
            << " median_bandwidth_gbps=" << median
            << " variance_gbps2=" << variance
            << " forwarding_batch_available_calls=" << forwarding_availability_calls
            << " forwarding_batches_issued=" << forwarding_batches_issued
            << " forwarding_batch_available_avg_us_per_call="
                << forwarding_availability_avg_us_per_call
            << " forwarding_batch_available_avg_us_per_batch="
                << forwarding_availability_avg_us_per_batch;

    if (config_.rdma_bandwidth_summary_dir.empty()) {
        RDMA_PROXY_LOG_INFO(summary.str());
        return;
    }

    std::filesystem::path dir(config_.rdma_bandwidth_summary_dir);
    std::filesystem::create_directories(dir);
    const auto filename = "rdma_bandwidth_summary_rank_" +
        std::to_string(config_.node_rank) + "_gpu_" +
        std::to_string(config_.local_gpu_index) + ".txt";
    const auto path = dir / filename;
    const auto tmp_path = path.string() + ".tmp." + std::to_string(current_process_id());

    {
        std::ofstream out(tmp_path, std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to write RDMA bandwidth summary file: " + tmp_path);
        }
        out << summary.str() << '\n';
        out.close();
        if (!out) {
            throw std::runtime_error("failed to flush RDMA bandwidth summary file: " + tmp_path);
        }
    }
    std::filesystem::rename(tmp_path, path);
    RDMA_PROXY_LOG_DEBUG("wrote RDMA bandwidth summary local_rank=", config_.node_rank,
                         " local_gpu=", config_.local_gpu_index,
                         " path=", path.string());
}

void Proxy::report_iteration(
    uint64_t iteration,
    std::chrono::steady_clock::time_point start,
    double seconds,
    std::size_t total_bytes,
    const std::vector<std::vector<QPCompletionBaseline>>& baselines,
    const std::vector<IterationAssignment>& assignments,
    std::size_t verification_errors,
    std::size_t validation_errors) {
    const double gbps = seconds > 0.0 ? (static_cast<double>(total_bytes) * 8.0 / seconds / 1.0e9) : 0.0;
    const double latency_us = seconds * 1.0e6;
    rdma_iteration_bandwidth_gbps_.push_back(gbps);

    RDMA_PROXY_LOG_INFO("iteration=", iteration,
                        " local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " total_bytes=", total_bytes,
                        " elapsed_us=", static_cast<uint64_t>(latency_us),
                        " bandwidth_gbps=", std::fixed, std::setprecision(3), gbps,
                        " immediate_mismatches=", verification_errors,
                        " validation_errors=", validation_errors);

    if (config_.log_marker_wait_reports) {
        for (std::size_t peer_index = 0; peer_index < peers_.size(); ++peer_index) {
            const auto& peer = peers_[peer_index];
            const auto& assignment = assignments[peer_index];
            auto send_payload_done_time = std::chrono::steady_clock::time_point{};
            auto recv_payload_done_time = std::chrono::steady_clock::time_point{};
            auto send_marker_done_time = std::chrono::steady_clock::time_point{};
            auto recv_marker_done_time = std::chrono::steady_clock::time_point{};
            std::size_t send_payload_completions = 0;
            std::size_t recv_payload_completions = 0;

            for (std::size_t q = 0; q < peer.workers.size(); ++q) {
                const auto& worker = peer.workers[q];
                const auto send_delta = worker->send_completions() - baselines[peer_index][q].sends;
                const auto recv_delta = worker->recv_completions() - baselines[peer_index][q].recvs;
                send_payload_completions += static_cast<std::size_t>(send_delta);
                recv_payload_completions += static_cast<std::size_t>(recv_delta);

                if (assignment.expected_send_completions_by_qp[q] > 0) {
                    send_payload_done_time = std::max(
                        send_payload_done_time, worker->latest_send_completion_time());
                }
                if (recv_delta > 0) {
                    recv_payload_done_time = std::max(
                        recv_payload_done_time, worker->latest_recv_completion_time());
                }
                if (worker->send_marker_completions() > baselines[peer_index][q].send_markers) {
                    send_marker_done_time = std::max(
                        send_marker_done_time, worker->latest_send_marker_time());
                }
                if (worker->recv_marker_completions() > baselines[peer_index][q].recv_markers) {
                    recv_marker_done_time = std::max(
                        recv_marker_done_time, worker->latest_recv_marker_time());
                }
            }

            RDMA_PROXY_LOG_INFO("marker_wait_report iteration=", iteration,
                                " local_rank=", config_.node_rank,
                                " local_gpu=", config_.local_gpu_index,
                                " peer=", peer.peer_rank,
                                " remote_rank=", peer.peer_rank,
                                " remote_gpu=", peer.remote_gpu_index,
                                " send_payload_completions=", send_payload_completions,
                                " recv_payload_completions=", recv_payload_completions,
                                " send_payload_done_elapsed_us=",
                                    elapsed_us_since(send_payload_done_time, start),
                                " send_marker_done_elapsed_us=",
                                    elapsed_us_since(send_marker_done_time, start),
                                " send_marker_wait_after_payload_us=",
                                    nonnegative_delta_us(send_payload_done_time, send_marker_done_time),
                                " recv_payload_done_elapsed_us=",
                                    elapsed_us_since(recv_payload_done_time, start),
                                " recv_marker_done_elapsed_us=",
                                    elapsed_us_since(recv_marker_done_time, start),
                                " recv_marker_wait_after_payload_us=",
                                    nonnegative_delta_us(recv_payload_done_time, recv_marker_done_time));
        }
    }

    if (config_.log_qp_reports) {
        for (std::size_t peer_index = 0; peer_index < peers_.size(); ++peer_index) {
            const auto& peer = peers_[peer_index];
            const auto& assignment = assignments[peer_index];
            for (std::size_t q = 0; q < peer.workers.size(); ++q) {
                const auto& worker = peer.workers[q];
                const auto local_qp = peer.qps[q]->local_info();
                const auto remote_qp = peer.qps[q]->remote_info();
                const auto send_payload_time =
                    assignment.expected_send_completions_by_qp[q] > 0 ?
                        worker->latest_send_completion_time() : std::chrono::steady_clock::time_point{};
                const auto recv_delta = worker->recv_completions() - baselines[peer_index][q].recvs;
                const auto recv_payload_time =
                    recv_delta > 0 ?
                        worker->latest_recv_completion_time() : std::chrono::steady_clock::time_point{};
                const auto send_marker_time = worker->latest_send_marker_time();
                const auto recv_marker_time = worker->latest_recv_marker_time();
                const auto send_payload_elapsed_us = elapsed_us_since(send_payload_time, start);
                const auto recv_payload_elapsed_us = elapsed_us_since(recv_payload_time, start);
                const auto send_marker_elapsed_us = elapsed_us_since(send_marker_time, start);
                const auto recv_marker_elapsed_us = elapsed_us_since(recv_marker_time, start);
                const auto marker_gap_us =
                    send_marker_elapsed_us >= 0 && recv_marker_elapsed_us >= 0 ?
                        recv_marker_elapsed_us - send_marker_elapsed_us :
                        0;
                const auto send_delta = worker->send_completions() - baselines[peer_index][q].sends;
                const auto send_marker_delta =
                    worker->send_marker_completions() - baselines[peer_index][q].send_markers;
                const auto recv_marker_delta =
                    worker->recv_marker_completions() - baselines[peer_index][q].recv_markers;
                const auto post_error_delta = worker->post_errors() - baselines[peer_index][q].post_errors;
                const auto cq_error_delta = worker->cq_errors() - baselines[peer_index][q].cq_errors;
                const auto unexpected_delta =
                    worker->unexpected_immediate_completions() - baselines[peer_index][q].unexpected_imms;
                const auto errors = post_error_delta + cq_error_delta + unexpected_delta;
                const double qp_gbps = seconds > 0.0 ?
                    (static_cast<double>(assignment.bytes_by_qp[q]) * 8.0 / seconds / 1.0e9) : 0.0;
                RDMA_PROXY_LOG_INFO("qp_report iteration=", iteration,
                                    " local_rank=", config_.node_rank,
                                    " local_gpu=", config_.local_gpu_index,
                                    " peer=", peer.peer_rank,
                                    " remote_rank=", peer.peer_rank,
                                    " remote_gpu=", peer.remote_gpu_index,
                                    " qp=", q,
                                    " local_qpn=", local_qp.qp_num,
                                    " remote_qpn=", remote_qp.qp_num,
                                    " local_lid=", local_qp.lid,
                                    " remote_lid=", remote_qp.lid,
                                    " local_psn=", local_qp.psn,
                                    " remote_psn=", remote_qp.psn,
                                    " bytes=", assignment.bytes_by_qp[q],
                                    " assigned_chunks=", assignment.chunks_by_qp[q],
                                    " elapsed_us=", static_cast<uint64_t>(latency_us),
                                    " send_payload_done_elapsed_us=", send_payload_elapsed_us,
                                    " send_marker_elapsed_us=", send_marker_elapsed_us,
                                    " send_marker_wait_after_payload_us=",
                                        nonnegative_delta_us(send_payload_time, send_marker_time),
                                    " recv_payload_done_elapsed_us=", recv_payload_elapsed_us,
                                    " recv_marker_elapsed_us=", recv_marker_elapsed_us,
                                    " recv_marker_wait_after_payload_us=",
                                        nonnegative_delta_us(recv_payload_time, recv_marker_time),
                                    " marker_gap_us=", marker_gap_us,
                                    " bandwidth_gbps=", std::fixed, std::setprecision(3), qp_gbps,
                                    " send_completions=", send_delta,
                                    " expected_data_send_completions=",
                                        assignment.expected_send_completions_by_qp[q],
                                    " recv_immediate_completions=", recv_delta,
                                    " send_marker_completions=", send_marker_delta,
                                    " recv_marker_completions=", recv_marker_delta,
                                    " post_errors=", post_error_delta,
                                    " cq_errors=", cq_error_delta,
                                    " unexpected_immediates=", unexpected_delta,
                                    " errors=", errors);
            }
        }
    }
}

}  // namespace rdma_proxy
