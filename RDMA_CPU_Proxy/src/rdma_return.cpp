#include "proxy.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace rdma_proxy {
namespace {
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::runtime_error("RDMA return byte count overflow");
    return a * b;
}
std::vector<std::vector<ChunkDescriptor>> return_chunks(
    std::size_t rows, std::size_t width, std::size_t tile, std::size_t group_tiles,
    std::size_t element, std::size_t chunk_bytes) {
    const auto tiles = width / tile;
    std::vector<std::vector<ChunkDescriptor>> result((tiles + group_tiles - 1) / group_tiles);
    const auto tile_bytes = product(product(rows, tile), element);
    for (std::size_t g = 0; g < result.size(); ++g) {
        const auto first = g * group_tiles;
        const auto begin = product(first, tile_bytes);
        const auto bytes = product(std::min(group_tiles, tiles - first), tile_bytes);
        for (std::size_t done = 0; done < bytes;) {
            ChunkDescriptor chunk;
            chunk.chunk_index = result[g].size();
            chunk.src_offset_bytes = chunk.dst_offset_bytes = begin + done;
            chunk.length_bytes = std::min(chunk_bytes, bytes - done);
            result[g].push_back(chunk);
            done += chunk.length_bytes;
        }
    }
    return result;
}
}

const std::vector<std::size_t>& Proxy::outgoing_token_indices(int destination_node) const {
    if (!initialized_ || !config_.router_routing_enabled || destination_node < 0 ||
        destination_node >= config_.num_nodes)
        throw std::runtime_error("invalid outgoing token-index query");
    return router_routing_.token_indices_for_node(destination_node);
}

std::size_t Proxy::rdma_return_receive_rows(int peer) const {
    if (!initialized_ || peer == config_.node_rank || peer < 0 || peer >= config_.num_nodes ||
        !config_.router_routing_enabled)
        throw std::runtime_error("invalid RDMA return receive-row query");
    return router_routing_.token_indices_for_node(peer).size();
}

uintptr_t Proxy::configure_rdma_returns(
    const RdmaProxyNetworkReturnBuffer* buffers, std::size_t count,
    std::size_t width, std::size_t tile, std::size_t group_tiles, std::size_t element,
    std::size_t chunk_bytes, uint64_t timeout_ms) {
    if (!initialized_ || iteration_prepared_ || rdma_returns_ || !buffers || count != peers_.size() ||
        !count || !width || !tile || width % tile || !group_tiles || group_tiles > width / tile ||
        width != config_.token_dimension || element != dtype_size(config_.dtype) ||
        !chunk_bytes || chunk_bytes > std::numeric_limits<uint32_t>::max() ||
        !timeout_ms || !config_.num_iterations || config_.num_nodes >= (1 << 14) ||
        !config_.router_routing_enabled || !config_.router_local_input_staging_enabled)
        throw std::runtime_error("invalid RDMA return configuration");
    const auto groups = (width / tile + group_tiles - 1) / group_tiles;
    const auto total_groups = product(groups, config_.num_iterations);
    if (total_groups >= kReturnSequenceLimit) throw std::runtime_error("too many RDMA return groups");
    if (config_.rdma_qp_calibration_enabled && !rdma_qp_calibration_complete_)
        calibrate_rdma_qps();
    rdma_returns_ = std::make_unique<RdmaReturnState>();
    auto& state = *rdma_returns_;
    state.groups = groups;
    state.timeout_ms = timeout_ms;
    try {
        if (config_.mock_mode) {
            state.host_prefix = new int32_t(0);
            state.device_prefix = reinterpret_cast<uintptr_t>(state.host_prefix);
        } else {
#if RDMA_PROXY_HAVE_CUDA
            select_cuda_device_for_thread(config_.cuda_device_id, false);
            auto check = [](cudaError_t e) {
                if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            };
            check(cudaHostAlloc(reinterpret_cast<void**>(&state.host_prefix), sizeof(int32_t), cudaHostAllocMapped));
            void* ptr = nullptr;
            check(cudaHostGetDevicePointer(&ptr, state.host_prefix, 0));
            state.device_prefix = reinterpret_cast<uintptr_t>(ptr);
#else
            throw std::runtime_error("RDMA return publication requires CUDA");
#endif
        }
        __atomic_store_n(state.host_prefix, 0, __ATOMIC_RELEASE);
        // Register every buffer before any metadata exchange, sorted by rank so
        // multi-node pairwise TCP exchanges have the same acyclic ordering.
        std::vector<RdmaProxyNetworkReturnBuffer> sorted(buffers, buffers + count);
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.peer < b.peer; });
        int last = -1;
        for (const auto& b : sorted) {
            const auto it = std::find_if(peers_.begin(), peers_.end(), [&](const auto& p) { return p.peer_rank == b.peer; });
            if (it == peers_.end() || b.peer == last || !b.source || !b.destination ||
                b.source_rows != it->router_x4.size() || b.destination_rows != rdma_return_receive_rows(b.peer))
                throw std::runtime_error("RDMA return buffer identity/rows mismatch");
            last = b.peer;
            state.peers.emplace_back();
            auto& p = state.peers.back();
            p.peer_index = static_cast<std::size_t>(it - peers_.begin());
            const auto send_bytes = product(product(b.source_rows, width), element);
            const auto recv_bytes = product(product(b.destination_rows, width), element);
            p.source = rdma_context_.register_memory(reinterpret_cast<void*>(b.source),
                std::max(element, send_bytes), "rdma_return_source");
            p.destination = rdma_context_.register_memory(reinterpret_cast<void*>(b.destination),
                std::max(element, recv_bytes), "rdma_return_destination");
            p.chunks = return_chunks(b.source_rows, width, tile, group_tiles, element, chunk_bytes);
            for (auto& worker : it->workers) worker->configure_return_protocol(total_groups);
        }
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            const auto& b = sorted[i];
            auto& p = state.peers[i];
            const auto addr = std::find_if(config_.peers.begin(), config_.peers.end(),
                [&](const auto& a) { return a.node_rank == b.peer; });
            if (addr == config_.peers.end()) throw std::runtime_error("RDMA return peer address missing");
            std::ostringstream payload;
            payload << "rdma_return_v1 " << config_.node_rank << ' ' << config_.local_gpu_index << ' '
                    << width << ' ' << tile << ' ' << group_tiles << ' ' << element << ' '
                    << chunk_bytes << ' ' << config_.num_iterations << ' '
                    << b.source_rows << ' ' << b.destination_rows << ' '
                    << p.destination.addr << ' ' << p.destination.rkey << ' ' << p.destination.length;
            std::istringstream remote(connection_manager_.exchange_control_message(*addr, payload.str(), timeout_ms));
            std::string version, trailing;
            int node = -1, gpu = -1;
            std::size_t rw = 0, rt = 0, rg = 0, re = 0, rc = 0, ri = 0, rs = 0, rr = 0;
            if (!(remote >> version >> node >> gpu >> rw >> rt >> rg >> re >> rc >> ri >> rs >> rr
                         >> p.remote.addr >> p.remote.rkey >> p.remote.length) || remote >> trailing ||
                version != "rdma_return_v1" || node != (config_.mock_mode ? config_.node_rank : b.peer) ||
                gpu != config_.local_gpu_index || rw != width || rt != tile || rg != group_tiles ||
                re != element || rc != chunk_bytes || ri != config_.num_iterations ||
                rs != b.destination_rows || rr != b.source_rows || !p.remote.addr ||
                p.remote.length != std::max(element, product(product(b.source_rows, width), element)))
                throw std::runtime_error("RDMA return remote metadata mismatch");
        }
    } catch (...) {
        release_rdma_returns();
        throw;
    }
    return state.device_prefix;
}

void Proxy::prepare_rdma_returns(uint64_t iteration) {
    if (!rdma_returns_) return;
    auto& state = *rdma_returns_;
    if (!state.complete || (state.active && iteration != state.iteration + 1) || (!state.active && iteration != 0))
        throw std::runtime_error("previous RDMA return iteration is not complete");
    __atomic_store_n(state.host_prefix, 0, __ATOMIC_RELEASE);
    state.iteration = iteration;
    state.active = true;
    state.complete = false;
    for (auto& p : state.peers) p.next_group = 0;
}

void Proxy::service_rdma_returns() const {
    if (!rdma_returns_ || !rdma_returns_->active || rdma_returns_->complete) return;
    auto& state = *rdma_returns_;
    const auto ready = __atomic_load_n(state.host_prefix, __ATOMIC_ACQUIRE);
    if (ready < 0 || static_cast<std::size_t>(ready) > state.groups)
        throw std::runtime_error("invalid GPU RDMA return-ready prefix");
    for (auto& p : state.peers) {
        auto& peer = peers_[p.peer_index];
        if (static_cast<std::size_t>(ready) < p.next_group)
            throw std::runtime_error("RDMA return-ready prefix regressed");
        while (p.next_group < static_cast<std::size_t>(ready)) {
            const auto sequence = state.iteration * state.groups + p.next_group;
            auto distributor = std::make_shared<DynamicChunkDistributor>(
                p.chunks[p.next_group], peer.peer_rank, peer.workers.size(),
                p.source.addr, p.source.lkey, p.remote.addr, p.remote.rkey, sequence);
            for (auto& worker : peer.workers) {
                SendTask task;
                task.distributor = distributor;
                worker->enqueue(std::move(task));
            }
            ++p.next_group;
        }
    }
}

void Proxy::finish_rdma_returns() const {
    if (!rdma_returns_) return;
    auto& state = *rdma_returns_;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(state.timeout_ms);
    const auto expected = (state.iteration + 1) * state.groups;
    for (;;) {
        service_rdma_returns();
        check_forwarding_error();
        bool complete = true;
        for (auto& p : state.peers) {
            if (p.next_group != state.groups) complete = false;
            for (const auto& worker : peers_[p.peer_index].workers) {
                if (worker->post_errors() || worker->cq_errors())
                    throw std::runtime_error("RDMA return worker failure: " + worker->last_error());
                if (worker->return_send_markers() != expected || worker->return_recv_markers() != expected)
                    complete = false;
            }
        }
        if (complete) break;
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("RDMA returns timed out");
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    if (gpudirect_rdma_writes_need_owner_flush(config_.cuda_device_id, config_.mock_mode))
        flush_gpudirect_rdma_writes(config_.cuda_device_id, config_.mock_mode);
    state.complete = true;
}

void Proxy::check_rdma_return(int peer, const std::string& sent, const std::string& received) const {
    if (!rdma_returns_ || !rdma_returns_->active || !rdma_returns_->complete ||
        sent.size() != 64 || received.size() != 64)
        throw std::runtime_error("invalid RDMA return checksum check");
    const auto addr = std::find_if(config_.peers.begin(), config_.peers.end(),
        [&](const auto& a) { return a.node_rank == peer; });
    if (addr == config_.peers.end()) throw std::runtime_error("RDMA return checksum peer missing");
    const auto prefix = "rdma_return_check " + std::to_string(rdma_returns_->iteration) + " ";
    const auto remote = connection_manager_.exchange_control_message(*addr, prefix + sent + " " + received,
                                                                     rdma_returns_->timeout_ms);
    if (remote != prefix + received + " " + sent)
        throw std::runtime_error("RDMA return SHA256 mismatch");
}

void Proxy::release_rdma_returns() {
    if (!rdma_returns_) return;
    for (auto& p : rdma_returns_->peers) {
        if (p.source.addr) rdma_context_.deregister_memory(p.source);
        if (p.destination.addr) rdma_context_.deregister_memory(p.destination);
    }
    if (config_.mock_mode) delete rdma_returns_->host_prefix;
#if RDMA_PROXY_HAVE_CUDA
    else if (rdma_returns_->host_prefix) cudaFreeHost(rdma_returns_->host_prefix);
#endif
    rdma_returns_.reset();
}
} // namespace rdma_proxy
