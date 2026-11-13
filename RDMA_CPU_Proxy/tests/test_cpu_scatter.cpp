#include "cuda_buffers.hpp"

#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rdma_proxy;
static void require(bool value) {
    if (!value) throw std::runtime_error("CPU scatter check failed");
}

template<class F> static void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}
struct alignas(16) SlabBlock { unsigned char bytes[16]; };

// Test-only access: production exposes no callbacks or thread-control API.
namespace rdma_proxy {
struct CpuScatterTestAccess {
    static void set_hook(CudaBuffers& buffers, std::function<void()> hook) {
        buffers.cpu_scatter_pack_test_hook_ = std::move(hook);
    }
    static void require_full_bundle(CudaBuffers& buffers) {
        std::lock_guard<std::mutex> lock(buffers.cpu_scatter_worker_mutex_);
        const auto groups = buffers.router_notification_publication_buffers_.n_groups_per_m_cluster;
        require(groups != 0);
        require(buffers.cpu_scatter_worker_consumed_rows_ % groups == 0);
        require(buffers.cpu_scatter_worker_committed_rows_ >=
                buffers.cpu_scatter_worker_consumed_rows_ + groups);
    }
    static void drain(CudaBuffers& buffers, std::size_t prefix) {
        std::unique_lock<std::mutex> lock(buffers.cpu_scatter_worker_mutex_);
        const auto groups = buffers.router_notification_publication_buffers_.n_groups_per_m_cluster;
        prefix = prefix / groups * groups;
        require(buffers.cpu_scatter_worker_cv_.wait_for(lock, std::chrono::seconds(5), [&] {
            return buffers.cpu_scatter_worker_error_ || buffers.cpu_scatter_worker_stop_ ||
                (!buffers.cpu_scatter_worker_busy_ && buffers.cpu_scatter_worker_consumed_rows_ >= prefix);
        }));
        if (buffers.cpu_scatter_worker_error_) std::rethrow_exception(buffers.cpu_scatter_worker_error_);
        require(!buffers.cpu_scatter_worker_stop_);
    }
    static void wait_cancelled(CudaBuffers& buffers) {
        std::unique_lock<std::mutex> lock(buffers.cpu_scatter_worker_mutex_);
        require(buffers.cpu_scatter_worker_cv_.wait_for(lock, std::chrono::seconds(5), [&] {
            return buffers.cpu_scatter_worker_cancelled_ || buffers.cpu_scatter_worker_stop_;
        }));
    }
    static void require_empty_epoch(CudaBuffers& buffers, uint64_t epoch) {
        std::lock_guard<std::mutex> lock(buffers.cpu_scatter_worker_mutex_);
        require(buffers.cpu_scatter_worker_epoch_ == epoch);
        require(!buffers.cpu_scatter_worker_busy_ && buffers.cpu_scatter_worker_consumed_rows_ == 0 &&
                buffers.cpu_scatter_worker_staged_rows_ == 0 && buffers.cpu_scatter_worker_committed_rows_ == 0);
    }
};
}

struct PackingGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool pause{false}, entered{false}, released{true}, fail{false};
    void arm(bool inject_error = false) {
        std::lock_guard<std::mutex> lock(mutex);
        pause = !inject_error;
        entered = false;
        released = false;
        fail = inject_error;
    }
    void operator()() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        cv.notify_all();
        if (fail) {
            fail = false;
            throw std::runtime_error("injected CPU scatter worker failure");
        }
        if (pause) {
            cv.wait(lock, [&] { return released; });
            pause = false;
        }
    }
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        require(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }));
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
    ~PackingGate() { release(); }
};
struct ReleasePackingGate {
    PackingGate& gate;
    ~ReleasePackingGate() { gate.release(); }
};
struct StopWorkerBeforeStorageRelease {
    std::unique_ptr<CudaBuffers>& buffers;
    PackingGate& gate;
    ~StopWorkerBeforeStorageRelease() {
        gate.release();
        if (buffers) {
            try { buffers->stop_cpu_scatter_worker(); }
            catch (...) { /* Preserve an existing test failure during cleanup. */ }
        }
    }
};

// publication_mode: legacy per-batch=0, deferred=1, dedicated CPU thread=2.
static void run(bool per_entry, uint64_t tile, bool cuda, int weight_mode,
                int publication_mode, bool memcpy_flag, int lifecycle = 0) {
    const bool deferred = publication_mode == 1;
    const bool threaded = publication_mode == 2;
    const bool single_upload = deferred || threaded;
    ProxyConfig config;
    config.mock_mode = !cuda;
    config.router_routing_enabled = true;
    config.router_weighted_contributions_enabled = weight_mode != 0;
    config.nvlink_forwarding_enabled = true;
    config.nvlink_forward_completion_notifications_enabled = true;
    config.nvlink_forward_notification_flush_per_entry_enabled = per_entry;
    config.nvlink_forward_notification_flag_update_mode = memcpy_flag ?
        NvlinkForwardNotificationFlagUpdateMode::kMemcpy : NvlinkForwardNotificationFlagUpdateMode::kStreamWrite;
    config.num_nodes = 1;
    config.num_gpus_per_node = 3;
    config.router_num_experts = 6;
    config.router_top_k = 2;
    config.num_tokens = 10;
    config.token_dimension = 256;
    config.dtype = DataType::kBF16;
    config.expert_gemm_m_tile = 2;
    config.expert_gemm_cluster_m = 1;
    config.expert_gemm_n_tile = 128;
    config.expert_gemm_dimension = 512;
    config.expert_gemm_max_swizzle_size = 2; // repeated N groups
    PackingGate packing_gate; // Must outlive every callback and worker join.
    auto buffers_owner = std::make_unique<CudaBuffers>(config);
    ReleasePackingGate release_before_buffers_cleanup{packing_gate};
    auto& buffers = *buffers_owner;
    buffers.initialize();
    const std::vector<std::vector<int32_t>> indices{{2,5,9,5,9}, {1,7,1,7}, {}};
    const std::vector<std::vector<int32_t>> offsets{{0,3,5}, {0,2,4}, {0,0,0}};
    std::vector<std::vector<float>> weights{{0.125f, 0.25f, 0.0f, 0.75f, 1.0f},
                                             {0.375f, 0.625f, 0.5f, 0.875f}, {}};
    if (weight_mode == 2) for (auto& source : weights) std::fill(source.begin(), source.end(), 1.0f);
    for (int source = 0; source < 3; ++source) {
        RouterExpertMetadata m;
        m.destination_node_rank = 0;
        m.destination_gpu_index = 0;
        m.source_node_rank = 0;
        m.source_gpu_index = source;
        m.num_nodes = 1;
        m.num_gpus_per_node = 3;
        m.num_experts = 6;
        m.experts_per_gpu = 2;
        m.first_global_expert = 0;
        m.num_tokens = 10;
        m.expert_token_indices = indices[source];
        m.expert_offsets = offsets[source];
        m.top_k = config.router_top_k;
        m.weighted_contributions = weight_mode != 0;
        if (weight_mode) m.expert_route_weights = weights[source];
        buffers.install_expert_metadata(m);
    }
    for (int source = 0; source < 3; ++source) {
        const auto& buffer = buffers.nvlink_receive_buffer_for_source(0, source);
        if (!weight_mode) require(buffer.expert_route_weights_device.ptr == nullptr);
        else {
            require(buffer.expert_route_weights_device.bytes == buffer.expert_token_indices_device.bytes);
            std::vector<float> copied(buffer.expert_route_weights_device.bytes / sizeof(float));
            if (!cuda) std::copy_n(static_cast<const float*>(buffer.expert_route_weights_device.ptr),
                                  copied.size(), copied.begin());
#if RDMA_PROXY_HAVE_CUDA
            else require(cudaMemcpy(copied.data(), buffer.expert_route_weights_device.ptr,
                                    buffer.expert_route_weights_device.bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
#endif
            require(std::equal(weights[source].begin(), weights[source].end(), copied.begin()));
            require(std::all_of(copied.begin() + weights[source].size(), copied.end(),
                                [](float value) { return value == 0; }));
        }
    }
    const std::size_t columns = tile ? 2 : 1;
    constexpr std::size_t routes = 9;
    std::vector<int64_t> scatter(routes*columns, -1), counts(3), inverse(30);
    std::vector<float> packed_weights(routes, -1.0f);
    const auto scatter_bytes = scatter.size() * sizeof(int64_t);
    const auto weight_offset = (scatter_bytes + 15) & ~std::size_t{15};
    const auto slab_bytes = weight_mode ? weight_offset + routes * sizeof(float) : scatter_bytes;
    // A deliberate 16-byte gap in legacy mode tests rejection of a misplaced
    // weight pointer without depending on allocator adjacency.
    const auto weight_gap = weight_mode && !single_upload ? std::size_t{16} : 0;
    const auto allocation_bytes = slab_bytes + weight_gap;
    std::vector<SlabBlock> slab((allocation_bytes + 15) / 16);
    void* compute_stream = create_cuda_stream(config.cuda_device_id, true, config.mock_mode);
    const auto compute_stream_address = cuda ? reinterpret_cast<uintptr_t>(compute_stream) : uintptr_t{1};
    void* device_weights = packed_weights.data();
    void* device_scatter = slab.data();
#if RDMA_PROXY_HAVE_CUDA
    if (cuda) {
        require(cudaMalloc(&device_scatter, allocation_bytes) == cudaSuccess);
    }
#endif
    if (weight_mode) device_weights = static_cast<unsigned char*>(device_scatter) + weight_offset + weight_gap;
    int32_t pairs[]{0,0,0,1,0,2};
    uint64_t bases[]{0x10000, 0x20000, 0x30000};
    RdmaProxyCpuScatterConfig setup{sizeof(setup), 3, routes, 10, 256, tile, 2,
        reinterpret_cast<uintptr_t>(device_scatter), pairs, bases, counts.data(), inverse.data()};
    auto bad = setup;
    bad.routes++;
    bool rejected = false;
    try { buffers.configure_cpu_scatter(bad); } catch (const std::exception&) { rejected = true; }
    require(rejected);
    buffers.configure_cpu_scatter(setup);
    if (weight_mode) {
        bool bad_count = false;
        try { buffers.configure_cpu_route_weights(reinterpret_cast<uintptr_t>(device_weights), routes + 1); }
        catch (const std::exception&) { bad_count = true; }
        require(bad_count);
        buffers.configure_cpu_route_weights(reinterpret_cast<uintptr_t>(device_weights), routes);
    }
    if (single_upload) {
        const auto configure = [&](std::size_t bytes) {
            if (threaded) buffers.configure_cpu_scatter_threaded(bytes);
            else buffers.configure_cpu_scatter_deferred(bytes);
        };
        rejects([&] { configure(slab_bytes - 1); });
        rejects([&] { configure(slab_bytes + 1); });
        configure(slab_bytes);
        rejects([&] { configure(slab_bytes); });
    } else if (weight_mode) {
        // A noncontiguous weight view cannot be passed as a single slab.
        rejects([&] { buffers.configure_cpu_scatter_deferred(slab_bytes); });
    }
    rejects([&] { buffers.wait_cpu_scatter_ready(0, compute_stream_address, 1); });
    require(counts == std::vector<int64_t>({3,2,0}));
    require(inverse == std::vector<int64_t>({2,5,9,-1,-1,-1,-1,-1,-1,-1,
                                          1,7,-1,-1,-1,-1,-1,-1,-1,-1,
                                          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1}));
    const auto& publication = buffers.router_notification_publication_buffers();
    if (threaded) CpuScatterTestAccess::set_hook(buffers, [&] {
        CpuScatterTestAccess::require_full_bundle(buffers);
        packing_gate();
    });
    // Declared after borrowed slab storage so assertion unwinding joins the
    // packer before any of that storage is destroyed.
    StopWorkerBeforeStorageRelease stop_before_slab_cleanup{buffers_owner, packing_gate};
    std::vector<int64_t> first_iteration;
    for (uint64_t iteration = 0; iteration < 2; ++iteration) {
        const auto packed_before = publication.cpu_scatter_packed_rows.load();
        const auto pack_ready_before = publication.cpu_scatter_pack_ready_publication_count.load();
        buffers.begin_router_notification_iteration(iteration);
        require(publication.cpu_scatter_packed_rows == packed_before);
        require(publication.cpu_scatter_pack_ready_publication_count == pack_ready_before);
        const auto copies_before = publication.cpu_scatter_copy_count.load();
        const auto bytes_before = publication.cpu_scatter_copy_bytes.load();
        std::vector<int32_t> partial_ready_prefixes;
        bool final_publication_seen = false;
        uint64_t final_pack_ready_count = 0;
        std::fill(scatter.begin(), scatter.end(), -1);
        if (!cuda) std::memset(device_scatter, 0xff, allocation_bytes);
#if RDMA_PROXY_HAVE_CUDA
        if (cuda) {
            require(cudaMemset(device_scatter, 0xff, allocation_bytes) == cudaSuccess);
            require(cudaDeviceSynchronize() == cudaSuccess);
        }
#endif
        std::future<void> waiter;
        std::vector<int64_t> consumer_scatter(scatter.size(), -1);
        if (single_upload) {
            rejects([&] { buffers.wait_cpu_scatter_ready(iteration, 0, 1); });
            rejects([&] { buffers.wait_cpu_scatter_ready(iteration, compute_stream_address, 0); });
            rejects([&] { buffers.wait_cpu_scatter_ready(iteration + 1, compute_stream_address, 1); });
            if (iteration) rejects([&] { buffers.wait_cpu_scatter_ready(iteration - 1, compute_stream_address, 1); });
            rejects([&] { buffers.wait_cpu_scatter_ready(iteration, compute_stream_address, 1); });
            waiter = std::async(std::launch::async, [&] {
                buffers.wait_cpu_scatter_ready(iteration, compute_stream_address, 5000);
#if RDMA_PROXY_HAVE_CUDA
                if (cuda) {
                    require(cudaMemcpyAsync(consumer_scatter.data(), device_scatter, scatter_bytes,
                        cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(compute_stream)) == cudaSuccess);
                    require(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(compute_stream)) == cudaSuccess);
                } else
#endif
                    std::memcpy(consumer_scatter.data(), device_scatter, scatter_bytes);
            });
            // No event has been recorded yet: a device-only wait here would incorrectly finish.
            require(waiter.wait_for(std::chrono::milliseconds(2)) == std::future_status::timeout);
        }
        if (threaded) packing_gate.arm();
        const auto check_published = [&]() {
            int32_t ready = 0;
            auto* table = static_cast<const int32_t*>(publication.device_table.ptr);
            std::vector<int32_t> table_copy;
            if (!cuda) {
                ready = *static_cast<const int32_t*>(publication.device_ready_rows.ptr);
                std::memcpy(scatter.data(), device_scatter, scatter_bytes);
                if (weight_mode) std::memcpy(packed_weights.data(), device_weights, routes * sizeof(float));
            }
#if RDMA_PROXY_HAVE_CUDA
            else {
                buffers.synchronize_router_notification_publication();
                table_copy.resize(publication.device_table.bytes / sizeof(int32_t));
                require(cudaMemcpy(&ready, publication.device_ready_rows.ptr, sizeof(ready),
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                require(cudaMemcpy(table_copy.data(), table, publication.device_table.bytes,
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                require(cudaMemcpy(scatter.data(), device_scatter, scatter.size()*sizeof(int64_t),
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                if (weight_mode)
                    require(cudaMemcpy(packed_weights.data(), device_weights, packed_weights.size()*sizeof(float),
                                       cudaMemcpyDeviceToHost) == cudaSuccess);
                table = table_copy.data();
            }
#endif
            require(ready == static_cast<int32_t>(
                buffers.router_computation_scheduler_state().published_rows));
            if (ready > 0 && static_cast<std::size_t>(ready) < publication.table_rows &&
                (partial_ready_prefixes.empty() || partial_ready_prefixes.back() != ready))
                partial_ready_prefixes.push_back(ready);
            if (single_upload && static_cast<std::size_t>(ready) != publication.table_rows) {
                if (deferred) {
                    // Deferred CPU work itself starts only after final gather.
                    require(publication.cpu_scatter_packed_rows == packed_before);
                    require(publication.cpu_scatter_pack_ready_publication_count == pack_ready_before);
                } else {
                    // The threaded packer has drained this partial prefix, but
                    // no upload is allowed until the final gather publication.
                    std::size_t covered = 0;
                    for (int r = 0; r < ready; r += publication.n_groups_per_m_cluster)
                        for (int source = 0; source < 3; ++source)
                            covered += table[r * publication.table_width + 3 + 2 * source] -
                                       table[r * publication.table_width + 2 + 2 * source];
                    require(covered > 0 && covered < routes);
                    require(publication.cpu_scatter_packed_rows == packed_before + covered);
                }
                require(publication.cpu_scatter_copy_count == copies_before);
                require(publication.cpu_scatter_copy_bytes == bytes_before);
                require(std::all_of(scatter.begin(), scatter.end(), [](int64_t x) { return x == -1; }));
                require(waiter.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
                return;
            }
            std::vector<int> seen(routes);
            std::size_t expected_packed = 0;
            for (int r = 0; r < ready; r += publication.n_groups_per_m_cluster) {
                auto* row = table + r * publication.table_width;
                std::size_t packed = 0;
                for (int s = 0; s < 3; ++s) packed += row[2+2*s];
                for (int s = 0; s < 3; ++s) {
                    for (int i = row[2+2*s]; i < row[3+2*s]; ++i, ++packed) {
                        require(packed < routes && !seen[packed]++);
                        ++expected_packed;
                        if (weight_mode) require(packed_weights[packed] == weights[s][i]);
                        const auto input = indices[s][i];
                        const auto begin = inverse.begin() + s*10;
                        const auto compact = std::find(begin, begin + counts[s], input) - begin;
                        require(compact < counts[s]);
                        require(scatter[packed*columns] == static_cast<int64_t>(
                            bases[s] + compact*(tile ? tile : 256)*2));
                        if (tile) require(scatter[packed*columns+1] == counts[s]*tile*2);
                    }
                }
            }
            // Repeated N groups share each M batch's packing; no routed row is
            // repacked either in legacy mode or during the final deferred walk.
            require(publication.cpu_scatter_packed_rows == packed_before + expected_packed);
            if (static_cast<std::size_t>(ready) == publication.table_rows)
                require(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
        };
        // Chunked arrivals cause several partial gather publications before the
        // final one. Reversing source priority changes packed row positions.
        // Unused tail rows and the empty source still notify after finalization.
        struct Arrival { int source; std::size_t begin; std::size_t count; };
        const auto arrivals = iteration == 0 ?
            std::vector<Arrival>{{0,0,6}, {1,0,2}, {1,2,6}, {0,6,4}, {1,8,2}, {2,0,10}} :
            std::vector<Arrival>{{1,0,8}, {0,0,3}, {0,3,3}, {0,6,4}, {1,8,2}, {2,0,10}};
        for (std::size_t arrival_index = 0; arrival_index < arrivals.size(); ++arrival_index) {
            const auto& arrival = arrivals[arrival_index];
            buffers.process_router_notification_completion(
                0, arrival.source, iteration, arrival.begin, arrival.count);
            if (threaded && arrival_index == 0) {
                packing_gate.wait_entered();
                const auto blocked_prefix = buffers.router_computation_scheduler_state().published_rows;
                require(blocked_prefix > 0 && blocked_prefix < publication.table_rows);
                partial_ready_prefixes.push_back(static_cast<int32_t>(blocked_prefix));
                require(publication.cpu_scatter_packed_rows == packed_before);
                require(publication.cpu_scatter_copy_count == copies_before);
                // Notification processing must progress while the CPU packer is
                // deliberately blocked. A shared scheduler lock would deadlock.
                do {
                    require(++arrival_index < arrivals.size());
                    const auto next_arrival = arrivals[arrival_index];
                    auto notification = std::async(std::launch::async, [&buffers, iteration, next_arrival] {
                        buffers.process_router_notification_completion(
                            0, next_arrival.source, iteration, next_arrival.begin, next_arrival.count);
                    });
                    const bool advanced = notification.wait_for(std::chrono::seconds(5)) ==
                                          std::future_status::ready;
                    if (!advanced) packing_gate.release();
                    require(advanced);
                    notification.get();
                } while (buffers.router_computation_scheduler_state().published_rows == blocked_prefix);
                require(buffers.router_computation_scheduler_state().published_rows < publication.table_rows);
                require(publication.cpu_scatter_packed_rows == packed_before);
                require(publication.cpu_scatter_copy_count == copies_before);
                packing_gate.release();
            }
            if (threaded)
                CpuScatterTestAccess::drain(buffers, buffers.router_computation_scheduler_state().published_rows);
            if (deferred && !final_publication_seen &&
                buffers.router_computation_scheduler_state().published_rows == publication.table_rows) {
                // Snapshot is taken at the start of CPU packing. Matching the
                // FINAL ready-publication counter proves packing starts after
                // that submission, not immediately before it in the same call.
                final_pack_ready_count = publication.ready_rows_publication_count;
                require(publication.cpu_scatter_pack_ready_publication_count == final_pack_ready_count);
                require(final_pack_ready_count > pack_ready_before);
                final_publication_seen = true;
            }
            if (single_upload && waiter.valid() &&
                buffers.router_computation_scheduler_state().published_rows == publication.table_rows) {
                // Join the event-dependent consumer BEFORE any publication-stream sync below.
                waiter.get();
                require(std::all_of(consumer_scatter.begin(), consumer_scatter.end(), [](int64_t x) { return x != -1; }));
            }
            check_published();
            if (final_publication_seen)
                require(publication.cpu_scatter_pack_ready_publication_count == final_pack_ready_count);
            if (buffers.router_computation_scheduler_state().published_rows < publication.table_rows) {
                const auto packed_before_flush = publication.cpu_scatter_packed_rows.load();
                buffers.flush_router_notification_publication();
                require(publication.cpu_scatter_packed_rows == packed_before_flush);
                check_published();
            }
        }
        require(buffers.router_computation_scheduler_state().published_rows == publication.table_rows);
        require(partial_ready_prefixes.size() >= 2);
        require(publication.cpu_scatter_packed_rows == packed_before + routes);
        if (single_upload) {
            if (deferred) require(final_publication_seen);
            require(publication.cpu_scatter_copy_count == copies_before + 1);
            require(publication.cpu_scatter_copy_bytes == bytes_before + slab_bytes);
            require(consumer_scatter == scatter);
            buffers.wait_cpu_scatter_ready(iteration, compute_stream_address, 1);
            buffers.flush_router_notification_publication();
            require(publication.cpu_scatter_copy_count == copies_before + 1);
            if (deferred)
                require(publication.cpu_scatter_pack_ready_publication_count == final_pack_ready_count);
        } else {
            require(publication.cpu_scatter_copy_count > copies_before + 1);
            require(publication.cpu_scatter_copy_bytes == bytes_before + scatter_bytes +
                (weight_mode ? routes * sizeof(float) : 0));
        }
        // Repeated final flushes must neither upload nor pack a second time.
        const auto final_copies = publication.cpu_scatter_copy_count.load();
        for (int repeat = 0; repeat < 2; ++repeat) {
            buffers.flush_router_notification_publication();
            require(publication.cpu_scatter_packed_rows == packed_before + routes);
            require(publication.cpu_scatter_copy_count == final_copies);
            if (deferred)
                require(publication.cpu_scatter_pack_ready_publication_count == final_pack_ready_count);
        }
        if (!iteration) first_iteration = scatter;
        else require(scatter != first_iteration);
    }
    if (single_upload) {
        buffers.begin_router_notification_iteration(2);
        auto changed_generation = std::async(std::launch::async, [&] {
            try { buffers.wait_cpu_scatter_ready(2, compute_stream_address, 5000); }
            catch (const std::runtime_error&) { return true; }
            return false;
        });
        require(changed_generation.wait_for(std::chrono::milliseconds(2)) == std::future_status::timeout);
        buffers.begin_router_notification_iteration(3);
        require(changed_generation.get());
    }
    if (threaded && lifecycle == 1) {
        require(!cuda);
        buffers.begin_router_notification_iteration(4);
        packing_gate.arm();
        buffers.process_router_notification_completion(0, 0, 4, 0, 6);
        packing_gate.wait_entered();
        const auto copies_before_reset = publication.cpu_scatter_copy_count.load();
        auto resetting = std::async(std::launch::async, [&] {
            buffers.begin_router_notification_iteration(5);
        });
        ReleasePackingGate release_before_reset_future_cleanup{packing_gate};
        // Reset has requested cancellation, but cannot clear host descriptors or
        // reuse the slab until the blocked worker relinquishes its busy lease.
        CpuScatterTestAccess::wait_cancelled(buffers);
        const bool reset_blocked = resetting.wait_for(std::chrono::milliseconds(0)) ==
                                   std::future_status::timeout;
        packing_gate.release();
        require(reset_blocked);
        require(resetting.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        resetting.get();
        CpuScatterTestAccess::require_empty_epoch(buffers, 5);
        require(publication.cpu_scatter_copy_count == copies_before_reset);
        rejects([&] { buffers.wait_cpu_scatter_ready(4, compute_stream_address, 1); });
        // A fresh epoch remains usable after cancelling an in-flight partial pack.
        for (int source : {1, 0, 2})
            buffers.process_router_notification_completion(0, source, 5, 0, 10);
        buffers.wait_cpu_scatter_ready(5, compute_stream_address, 5000);
        require(publication.cpu_scatter_copy_count == copies_before_reset + 1);

        buffers.begin_router_notification_iteration(6);
        packing_gate.arm(true);
        auto failed_waiter = std::async(std::launch::async, [&] {
            try { buffers.wait_cpu_scatter_ready(6, compute_stream_address, 5000); }
            catch (const std::runtime_error& error) {
                return std::string(error.what()).find("injected CPU scatter worker failure") != std::string::npos;
            }
            return false;
        });
        require(failed_waiter.wait_for(std::chrono::milliseconds(2)) == std::future_status::timeout);
        buffers.process_router_notification_completion(0, 0, 6, 0, 6);
        require(failed_waiter.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        require(failed_waiter.get());
        require(publication.cpu_scatter_copy_count == copies_before_reset + 1);
        // Failed worker teardown still joins and never publishes stale metadata.
        buffers.stop_cpu_scatter_worker();
    }
    if (threaded && lifecycle == 2) {
        require(!cuda);
        buffers.begin_router_notification_iteration(4);
        packing_gate.arm();
        buffers.process_router_notification_completion(0, 0, 4, 0, 6);
        packing_gate.wait_entered();
        std::promise<void> destructor_started;
        auto started = destructor_started.get_future();
        auto destroying = std::async(std::launch::async, [&] {
            destructor_started.set_value();
            buffers_owner.reset();
        });
        ReleasePackingGate release_before_destructor_future_cleanup{packing_gate};
        started.get();
        const bool destructor_blocked = destroying.wait_for(std::chrono::milliseconds(20)) ==
                                        std::future_status::timeout;
        packing_gate.release();
        require(destructor_blocked);
        require(destroying.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        destroying.get();
        require(!buffers_owner);
        // The callback's captured gate is still alive here; successful destructor
        // completion establishes that no worker can use it after this scope.
    }
#if RDMA_PROXY_HAVE_CUDA
    if (cuda) {
        buffers.synchronize_router_notification_publication();
        buffers.stop_cpu_scatter_worker();
        require(cudaFree(device_scatter) == cudaSuccess);
    }
#endif
    destroy_cuda_stream(compute_stream, config.mock_mode);
}
int main(int argc, char** argv) {
    const bool cuda = argc == 2 && std::string(argv[1]) == "--cuda";
    if (cuda) {
#if RDMA_PROXY_HAVE_CUDA
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
#else
        return 77;
#endif
    }
    for (bool per_entry : {false, true}) for (uint64_t tile : {0, 128})
        for (int weights : {0, 1, 2}) for (int publication_mode : {0, 1, 2})
            for (bool memcpy_flag : {false, true})
                run(per_entry, tile, cuda, weights, publication_mode, memcpy_flag);
    if (!cuda) {
        run(false, 128, false, 1, 2, false, 1); // busy reset and worker failure
        run(false, 128, false, 1, 2, false, 2); // destructor joins a busy worker
    }
    std::cout << "test_cpu_scatter passed\n";
}
