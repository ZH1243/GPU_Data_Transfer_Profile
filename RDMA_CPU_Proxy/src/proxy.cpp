#include "proxy.hpp"

#include "logging.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
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
    char padding[40]{};
};

namespace {

constexpr uint64_t kLocalIterationSyncMagic = 0x52444d415053594eULL;  // "RDMAPSyn"
constexpr uint32_t kLocalIterationSyncVersion = 1;

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

}  // namespace

Proxy::Proxy(ProxyConfig config)
    : config_(std::move(config)),
      cuda_buffers_(config_),
      rdma_context_(config_),
      connection_manager_(config_) {}

Proxy::~Proxy() {
    shutdown();
}

void Proxy::initialize() {
    if (initialized_) return;
    RDMA_PROXY_LOG_INFO("initializing proxy: ", config_summary(config_));

    initialize_local_iteration_sync();
    cuda_buffers_.initialize();
    rdma_context_.initialize();

    for (auto& peer_buffers : cuda_buffers_.peer_buffers()) {
        setup_peer(peer_buffers);
    }
    start_forwarding_thread();
    initialized_ = true;
}

void Proxy::run_once() {
    if (!initialized_) throw std::runtime_error("proxy is not initialized");
    run_iteration(0);
    synchronize_iteration(0);
}

void Proxy::run() {
    if (!initialized_) throw std::runtime_error("proxy is not initialized");
    for (uint64_t iteration = 0;
         config_.num_iterations == 0 || iteration < static_cast<uint64_t>(config_.num_iterations);
         ++iteration) {
        run_iteration(iteration);
        synchronize_iteration(iteration);
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

    for (std::size_t q = 0; q < peer.qps.size(); ++q) {
        peer.qps[q]->connect(remote_info.qps[q]);
        auto worker = std::make_unique<QPWorker>(
            *peer.qps[q],
            config_.completion_poll_batch_size,
            config_.max_in_flight_chunks_per_qp);
        worker->configure_expected_chunks(make_chunks().size());
        worker->post_initial_receives(config_.recv_queue_depth);
        worker->start();
        peer.workers.emplace_back(std::move(worker));
    }

    synchronize_peer_ready(peer_addr, peer);
    RDMA_PROXY_LOG_INFO("peer ", buffers.peer_rank, " initialized with ", peer.qps.size(), " RC QPs");
    peers_.push_back(std::move(peer));
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
    if (!config_.local_iteration_sync_enabled || config_.num_gpus_per_node <= 1) return;
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

void Proxy::run_iteration(uint64_t iteration) {
    const auto chunks = make_chunks();
    const auto bytes_per_peer = cuda_buffers_.token_buffer_bytes();

    fill_iteration_send_buffers(iteration);

    std::vector<std::vector<QPCompletionBaseline>> baselines;
    baselines.reserve(peers_.size());
    for (const auto& peer : peers_) {
        baselines.push_back(capture_baselines(peer, chunks));
    }
    synchronize_iteration_start(iteration);

    const auto start = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<DynamicChunkDistributor>> dispatchers(peers_.size());
    if (config_.sequential_peer_transfers) {
        const auto order = sequential_peer_order();
        for (const auto peer_index : order) {
            dispatchers[peer_index] = enqueue_chunks(
                peers_[peer_index], cuda_buffers_.peer_buffers()[peer_index], chunks);
            wait_for_outgoing_transfer(peers_[peer_index], baselines[peer_index], dispatchers[peer_index]);
        }
        for (std::size_t i = 0; i < peers_.size(); ++i) {
            wait_for_iteration(peers_[i], baselines[i], dispatchers[i]);
        }
    } else {
        for (std::size_t i = 0; i < peers_.size(); ++i) {
            dispatchers[i] = enqueue_chunks(peers_[i], cuda_buffers_.peer_buffers()[i], chunks);
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
        verification_errors += verify_immediates(peers_[i], chunks, baselines[i], assignments[i], iteration);
    }
    const auto validation_errors = validate_received_data(iteration);

    report_iteration(
        iteration, start, seconds, bytes_per_peer, baselines, assignments,
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

std::vector<ChunkDescriptor> Proxy::make_chunks() const {
    return compute_chunks(
        config_.num_tokens,
        config_.token_dimension,
        dtype_size(config_.dtype),
        config_.tokens_per_chunk,
        config_.num_qps_per_peer);
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
    publish_local_nvlink_receive_buffers();
    prepare_forwarding_destinations();
    if (!config_.nvlink_forward_use_round_robin) {
        prepare_forwarding_routing_tables();
    }
    for (const auto& dst : forwarding_destinations_) {
        enable_cuda_peer_access(config_.cuda_device_id, dst.cuda_device_id, config_.mock_mode);
    }

    {
        std::lock_guard<std::mutex> lock(forwarding_mutex_);
        forwarding_next_batch_by_peer_.assign(peers_.size(), 0);
        forwarding_iteration_stats_.clear();
        forwarding_error_.clear();
    }
    forwarding_stop_.store(false);
    forwarding_thread_ = std::thread(&Proxy::forwarding_loop, this);
    RDMA_PROXY_LOG_INFO("NVLink forwarding thread started local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " threshold_tokens=", config_.nvlink_forward_threshold_tokens,
                        " chunk_tokens=", config_.nvlink_forward_chunk_tokens,
                        " use_batch_api=", config_.nvlink_forward_use_batch_api ? "true" : "false");
}

void Proxy::stop_forwarding_thread() {
    forwarding_stop_.store(true);
    if (forwarding_thread_.joinable()) forwarding_thread_.join();
    for (auto& dst : forwarding_destinations_) {
        if (dst.imported_cuda_ipc) {
            close_cuda_ipc_memory_handle(dst.ptr, config_.mock_mode);
        }
        dst.ptr = nullptr;
        dst.imported_cuda_ipc = false;
    }
    forwarding_destinations_.clear();
    forwarding_routing_tables_by_peer_.clear();
    destroy_cuda_stream(forwarding_stream_, config_.mock_mode);
    forwarding_stream_ = nullptr;
}

void Proxy::prepare_forwarding_routing_tables() {
    forwarding_routing_tables_by_peer_.clear();
    forwarding_routing_tables_by_peer_.reserve(peers_.size());
    std::mt19937_64 rng(config_.nvlink_routing_seed);
    std::bernoulli_distribution route(config_.nvlink_routing_probability);

    for (const auto& peer : peers_) {
        std::vector<uint8_t> table(config_.num_tokens, 0);
        for (auto& row : table) {
            for (std::size_t column = 0; column < 8; ++column) {
                if (route(rng)) {
                    row = static_cast<uint8_t>(row | routing_column_mask(column));
                }
            }
        }

        std::sort(table.begin(), table.end(), std::greater<uint8_t>());
        forwarding_routing_tables_by_peer_.push_back(std::move(table));
        RDMA_PROXY_LOG_INFO("generated NVLink routing table peer_rank=", peer.peer_rank,
                            " rows=", config_.num_tokens,
                            " active_columns=", active_routing_columns(config_.num_gpus_per_node),
                            " ignored_columns=", 8 - active_routing_columns(config_.num_gpus_per_node),
                            " probability=", config_.nvlink_routing_probability,
                            " seed=", config_.nvlink_routing_seed);
    }
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
        return ForwardDestinationState{
            it->gpu_index,
            it->cuda_device_id,
            reinterpret_cast<void*>(static_cast<uintptr_t>(it->buffer_addr)),
            it->buffer_bytes,
            false};
    }

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
        uint64_t mock_addr = 0;
        std::string ipc_handle;
        std::string key;
        bool found_source_buffer = false;
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
            } else if (key == "source_gpu") {
                int source_gpu = -1;
                std::string mock_key;
                std::string handle_key;
                uint64_t entry_mock_addr = 0;
                std::string entry_handle;
                in >> source_gpu >> mock_key >> entry_mock_addr >> handle_key >> entry_handle;
                if (!in || mock_key != "mock_addr" || handle_key != "ipc_handle") {
                    throw std::runtime_error("malformed NVLink source buffer entry in " + path);
                }
                if (source_gpu == config_.local_gpu_index) {
                    mock_addr = entry_mock_addr;
                    ipc_handle = entry_handle;
                    found_source_buffer = true;
                }
            } else {
                throw std::runtime_error("unknown key in NVLink exchange file " + path + ": " + key);
            }
        }

        if (!process_is_alive(exporter_pid)) {
            last_wait_reason = "exporter PID is not alive";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (node_rank != config_.node_rank || gpu_index != dst_gpu || cuda_device_id < 0 ||
            buffer_bytes == 0 || !found_source_buffer) {
            last_wait_reason = "metadata incomplete or mismatched";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        void* ptr = open_cuda_ipc_memory_handle(ipc_handle, mock_addr, config_.mock_mode);
        RDMA_PROXY_LOG_INFO("imported NVLink receive buffer local_rank=", config_.node_rank,
                            " src_gpu=", config_.local_gpu_index,
                            " dst_gpu=", dst_gpu,
                            " exporter_pid=", exporter_pid,
                            " cuda_device_id=", cuda_device_id,
                            " bytes=", buffer_bytes,
                            " ptr=", reinterpret_cast<uintptr_t>(ptr),
                            " path=", path);
        return ForwardDestinationState{dst_gpu, cuda_device_id, ptr, buffer_bytes, true};
    }

    throw std::runtime_error("timed out waiting for live NVLink exchange file: " + path +
                             " reason=" + last_wait_reason);
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
    const auto batch_end_token = batch_start_token + batch_tokens;
    std::vector<uint64_t> totals(chunks.size(), 0);
    for (const auto& worker : peer.workers) {
        const auto counts = worker->received_immediate_counts();
        for (std::size_t i = 0; i < counts.size() && i < totals.size(); ++i) {
            totals[i] += counts[i];
        }
    }

    bool overlaps_any_chunk = false;
    for (const auto& chunk : chunks) {
        const auto chunk_end_token = chunk.start_token + chunk.num_tokens;
        if (chunk.start_token >= batch_end_token || chunk_end_token <= batch_start_token) {
            continue;
        }
        overlaps_any_chunk = true;
        if (chunk.chunk_index >= totals.size() || totals[chunk.chunk_index] < required_count) {
            return false;
        }
    }
    return overlaps_any_chunk;
}

void Proxy::issue_forwarding_batch(
    const PeerState& peer,
    const PeerGpuBuffers& buffers,
    uint64_t iteration,
    std::size_t batch_index_in_iteration,
    std::size_t batch_start_token) {
    const auto token_bytes = config_.token_dimension * dtype_size(config_.dtype);
    const auto batch_timing_start = std::chrono::steady_clock::now();
    std::size_t batch_bytes = 0;
    const auto batch_tokens = config_.nvlink_forward_threshold_tokens;
    if (batch_start_token + batch_tokens > config_.num_tokens) {
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
    const auto peer_slot_offset = cuda_buffers_.token_buffer_bytes() * peer_slot;

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

            const auto token_offset = batch_start_token + chunk_index * config_.nvlink_forward_chunk_tokens;
            const auto source_byte_offset = token_offset * token_bytes;
            const auto destination_byte_offset = peer_slot_offset + source_byte_offset;
            const auto bytes = config_.nvlink_forward_chunk_tokens * token_bytes;
            if (source_byte_offset + bytes > buffers.recv.bytes ||
                destination_byte_offset + bytes > dst_it->bytes) {
                throw std::runtime_error("NVLink forwarding copy exceeds source or destination buffer size");
            }

            CudaForwardCopy copy;
            copy.src = static_cast<const char*>(buffers.recv.ptr) + source_byte_offset;
            copy.dst = static_cast<char*>(dst_it->ptr) + destination_byte_offset;
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
                                    " peer_slot_offset=", peer_slot_offset,
                                    " bytes=", bytes,
                                    " src_addr=", reinterpret_cast<uintptr_t>(copy.src),
                                    " dst_addr=", reinterpret_cast<uintptr_t>(copy.dst));
            }
            launch_cuda_forward_copy_batch_async(
                copy,
                forwarding_stream_,
                config_.nvlink_forward_use_batch_api,
                config_.mock_mode);
        }
    } else {
        if (peer_slot >= forwarding_routing_tables_by_peer_.size()) {
            throw std::runtime_error("missing NVLink routing table for peer buffer slot");
        }

        const auto& routing_table = forwarding_routing_tables_by_peer_[peer_slot];

        // Each routing row is one uint8_t for one received token. Columns
        // 0..num_gpus_per_node-2 map to local destination GPUs via
        // (local_gpu + 1 + column) % num_gpus_per_node. The remaining columns
        // are still generated and included in sorting, but are ignored for
        // forwarding because they do not represent unique peer GPUs.
        for (const auto& dst : forwarding_destinations_) {
            const auto route_column =
                routing_column_for_gpu(config_.local_gpu_index, dst.gpu_index, config_.num_gpus_per_node);
            const int mapped_gpu =
                routing_column_to_gpu(config_.local_gpu_index, route_column, config_.num_gpus_per_node);
            if (mapped_gpu != dst.gpu_index) {
                throw std::runtime_error("NVLink routing destination mapping mismatch");
            }

            std::vector<CudaForwardCopy> copies;
            copies.reserve(batch_tokens);
            const auto destination_base_offset = peer_slot_offset + batch_start_token * token_bytes;
            std::size_t routed_tokens = 0;
            std::size_t run_start_token = 0;
            std::size_t run_tokens = 0;
            auto flush_run = [&]() {
                if (run_tokens == 0) return;
                const auto source_byte_offset = run_start_token * token_bytes;
                const auto destination_byte_offset = destination_base_offset + routed_tokens * token_bytes;
                const auto bytes = run_tokens * token_bytes;
                if (source_byte_offset + bytes > buffers.recv.bytes ||
                    destination_byte_offset + bytes > dst.bytes) {
                    throw std::runtime_error("NVLink forwarding copy exceeds source or destination buffer size");
                }

                CudaForwardCopy copy;
                copy.src = static_cast<const char*>(buffers.recv.ptr) + source_byte_offset;
                copy.dst = static_cast<char*>(dst.ptr) + destination_byte_offset;
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
                                    " routed_tokens=", routed_tokens,
                                    " batch_entries=", copies.size(),
                                    " peer_slot_offset=", peer_slot_offset,
                                    " bytes=", bytes,
                                    " first_src_addr=", reinterpret_cast<uintptr_t>(copies.front().src),
                                    " first_dst_addr=", reinterpret_cast<uintptr_t>(copies.front().dst));
            }
            launch_cuda_forward_copy_batch_async(
                copies,
                forwarding_stream_,
                config_.nvlink_forward_use_batch_api,
                config_.mock_mode);
        }
    }
    if (config_.nvlink_forward_synchronize_batches) {
        synchronize_cuda_stream(forwarding_stream_, config_.mock_mode);
        const auto batch_timing_end = std::chrono::steady_clock::now();
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

void Proxy::forwarding_loop() {
    try {
        const auto chunks = make_chunks();
        const auto peer_order = nvlink_forward_peer_order();
        const auto batches_per_iteration =
            config_.num_tokens / config_.nvlink_forward_threshold_tokens;
        const bool finite_iterations = config_.num_iterations != 0;
        const auto total_batches = config_.num_iterations * batches_per_iteration;

        while (!forwarding_stop_.load()) {
            check_forwarding_error();
            bool progressed = false;
            for (const auto peer_index : peer_order) {
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
                    batch_in_iteration * config_.nvlink_forward_threshold_tokens;
                const auto required_count = iteration + 1;
                const auto& peer = peers_[peer_index];
                if (!forwarding_batch_available(
                        peer,
                        chunks,
                        batch_start_token,
                        config_.nvlink_forward_threshold_tokens,
                        required_count)) {
                    continue;
                }

                if (config_.nvlink_forward_log_batches) {
                    RDMA_PROXY_LOG_INFO("nvlink_forward_batch_ready iteration=", iteration,
                                        " local_rank=", config_.node_rank,
                                        " local_gpu=", config_.local_gpu_index,
                                        " peer_rank=", peer.peer_rank,
                                        " batch=", batch_in_iteration,
                                        " batch_start_token=", batch_start_token,
                                        " batch_tokens=", config_.nvlink_forward_threshold_tokens);
                }
                issue_forwarding_batch(
                    peer,
                    cuda_buffers_.peer_buffers()[peer_index],
                    iteration,
                    batch_in_iteration,
                    batch_start_token);
                {
                    std::lock_guard<std::mutex> lock(forwarding_mutex_);
                    forwarding_next_batch_by_peer_.at(peer_index) = next_batch + 1;
                }
                progressed = true;
            }
            if (!progressed) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
    } catch (const std::exception& e) {
        set_forwarding_error(e.what());
    }
}

void Proxy::wait_for_forwarding_iteration(uint64_t iteration) {
    if (!config_.nvlink_forwarding_enabled) return;
    const auto batches_per_iteration = config_.num_tokens / config_.nvlink_forward_threshold_tokens;
    const auto required_batches = static_cast<std::size_t>(iteration + 1) * batches_per_iteration;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    while (true) {
        check_forwarding_error();
        bool complete = true;
        {
            std::lock_guard<std::mutex> lock(forwarding_mutex_);
            for (const auto next_batch : forwarding_next_batch_by_peer_) {
                if (next_batch < required_batches) {
                    complete = false;
                    break;
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
                                    " batches_per_peer=", batches_per_iteration,
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
                                    " batches_per_peer=", batches_per_iteration);
            }
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            std::ostringstream out;
            out << "timed out waiting for NVLink forwarding iteration=" << iteration
                << " local_rank=" << config_.node_rank
                << " local_gpu=" << config_.local_gpu_index
                << " required_batches_per_peer=" << required_batches;
            std::lock_guard<std::mutex> lock(forwarding_mutex_);
            for (std::size_t i = 0; i < forwarding_next_batch_by_peer_.size(); ++i) {
                out << " peer" << peers_[i].peer_rank
                    << "=" << forwarding_next_batch_by_peer_[i];
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
    for (const auto& buffers : cuda_buffers_.peer_buffers()) {
        std::string error;
        const int expected_source = config_.mock_mode ? config_.node_rank : buffers.peer_rank;
        const int expected_destination = config_.mock_mode ? buffers.peer_rank : config_.node_rank;
        if (!cuda_buffers_.validate_recv_pattern(
                buffers.peer_rank, expected_source, expected_destination, iteration, &error)) {
            ++errors;
            RDMA_PROXY_LOG_ERROR("receive validation failed: ", error);
        }
    }
    return errors;
}

void Proxy::report_iteration(
    uint64_t iteration,
    std::chrono::steady_clock::time_point start,
    double seconds,
    std::size_t bytes_per_peer,
    const std::vector<std::vector<QPCompletionBaseline>>& baselines,
    const std::vector<IterationAssignment>& assignments,
    std::size_t verification_errors,
    std::size_t validation_errors) const {
    const auto total_bytes = bytes_per_peer * peers_.size();
    const double gbps = seconds > 0.0 ? (static_cast<double>(total_bytes) * 8.0 / seconds / 1.0e9) : 0.0;
    const double latency_us = seconds * 1.0e6;

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
