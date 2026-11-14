#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace rdma_proxy {
// Declared before proxy owners so callback storage outlives receiver shutdown.
struct EarlyReadyReceiverGate {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned calls{0}, pause_call{0}, fail_call{0};
    bool paused{false}, released{false};
    std::thread::id receiver;

    void operator()() {
        std::unique_lock<std::mutex> lock(mutex);
        receiver = std::this_thread::get_id();
        ++calls;
        if (calls == fail_call)
            throw std::runtime_error("injected early-ready receiver failure");
        if (calls == pause_call) {
            paused = true;
            changed.notify_all();
            if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return released; }))
                throw std::runtime_error("early-ready receiver test gate timed out");
        }
    }
    void wait_paused() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!changed.wait_for(lock, std::chrono::seconds(5), [&] { return paused; }))
            throw std::runtime_error("early-ready receiver did not reach test gate");
    }
    void verify(std::thread::id expected_receiver, unsigned expected_calls) {
        std::lock_guard<std::mutex> lock(mutex);
        if (calls != expected_calls || (calls && receiver != expected_receiver))
            throw std::runtime_error("early readiness was not published once by the notification receiver");
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        changed.notify_all();
    }
};

// Test access stays outside the public embedding ABI.
struct PreparedForwardingTestAccess {
    static void test_barrier_generations(bool ping_pong2 = false) {
        ProxyConfig config;
        config.mock_mode = true;
        config.num_gpus_per_node = 2;
        config.nvlink_forward_preparation_enabled = true;
        config.nvlink_forward_ping_pong2_enabled = ping_pong2;
        config.nvlink_forward_local_batch_sync_enabled = true;
        config.completion_timeout_ms = 5000;
        config.local_iteration_sync_run_id = "test_prepared_barrier_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        std::array<std::unique_ptr<Proxy>, 2> proxies;
        for (int gpu = 0; gpu < 2; ++gpu) {
            config.local_gpu_index = gpu;
            proxies[gpu] = std::make_unique<Proxy>(config);
            proxies[gpu]->initialize_local_iteration_sync();
        }
        std::atomic<uint64_t> progress[2][2]{};
        std::array<std::exception_ptr, 2> errors;
        auto run = [&](int gpu) {
            try {
                auto& proxy = *proxies[gpu];
                // No iteration barrier: deliberately let faster participants
                // change phases/iterations while peers still observe old ones.
                for (uint64_t iteration = 0; iteration < 100; ++iteration) {
                    for (int phase = 0; phase < 2; ++phase) {
                        const auto kind = static_cast<Proxy::LocalNvlinkBatchSyncPhase>(phase);
                        const uint64_t rounds = (iteration + gpu + phase) % 3 == 0 ? 0 :
                            gpu == phase ? 1 : 19;
                        for (uint64_t round = 1; round <= rounds; ++round) {
                            progress[gpu][phase].store(iteration * 100 + round);
                            if ((iteration + round + gpu) % 3 == 0) std::this_thread::yield();
                            const std::size_t chunks = gpu == 0 ? 32 : 7;
                            const auto result = ping_pong2 && phase == 1 ?
                                proxy.synchronize_prepared_nvlink_batch_start(
                                    kind, iteration, (round - 1) / 2 + 1, chunks, (round - 1) % 2) :
                                proxy.synchronize_local_nvlink_batch_start(
                                    kind, iteration, round - 1, round, chunks);
                            if (result != chunks) {
                                throw std::runtime_error("prepared barrier negotiated batch size");
                            }
                            if (progress[1 - gpu][phase].load() < iteration * 100 + round) {
                                throw std::runtime_error("prepared barrier passed before peer arrival");
                            }
                        }
                        progress[gpu][phase].store(iteration * 100 + 99);
                        proxy.mark_local_nvlink_batch_phase_complete(kind, iteration);
                    }
                }
            } catch (...) {
                errors[gpu] = std::current_exception();
                for (auto& proxy : proxies) proxy->forwarding_stop_.store(true);
            }
        };
        std::thread a(run, 0), b(run, 1);
        a.join(); b.join();
        for (auto error : errors) if (error) std::rethrow_exception(error);
    }

    static void test_ping_pong2_dispatch_order() {
        ProxyConfig config;
        config.mock_mode = true;
        config.nvlink_forward_preparation_enabled = true;
        config.nvlink_forward_ping_pong2_enabled = true;
        config.nvlink_forward_completion_notifications_enabled = true;
        config.nvlink_forward_notification_queue_depth = 1;
        config.completion_timeout_ms = 5000;
        Proxy proxy(config);
        proxy.initialize_prepared_forwarding();
        proxy.initialize_nvlink_forward_notification_dispatch();
        // Zero-record envelopes still occupy an ordered batch position. No
        // CUDA or shared-memory destination is needed for this dispatch test.
        proxy.enqueue_prepared_notifications(nullptr, 0, 1, 1);
        proxy.nvlink_forward_notification_dispatch_thread_ =
            std::thread(&Proxy::nvlink_forward_notification_dispatch_loop, &proxy);
        std::atomic<bool> started{false}, enqueued{false};
        std::exception_ptr error;
        std::thread producer([&] {
            try {
                started.store(true);
                proxy.enqueue_prepared_notifications(nullptr, 0, 1, 3);
                enqueued.store(true);
            } catch (...) { error = std::current_exception(); }
        });
        while (!started.load()) std::this_thread::yield();
        // B's queue must stay full until the missing A envelope arrives,
        // even though B has already completed its later batch.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const bool overtook = enqueued.load();
        proxy.enqueue_prepared_notifications(nullptr, 0, 0, 0);
        producer.join();
        if (error) std::rethrow_exception(error);
        proxy.enqueue_prepared_notifications(nullptr, 0, 0, 2);
        proxy.enqueue_prepared_notifications(nullptr, 0, 0, 4);
        // An odd iteration tail is followed by A again, not B. The global
        // dispatch sequence must survive this lane-parity reset.
        proxy.enqueue_prepared_notifications(nullptr, 0, 0, 5);
        proxy.stop_forwarding_thread();
        proxy.check_forwarding_error();
        if (overtook) throw std::runtime_error("dispatch overtook the missing batch");
    }

    static void fill_source(Proxy& proxy, uint64_t iteration) {
        for (auto& peer : proxy.cuda_buffers_.peer_buffers()) {
            auto* bytes = static_cast<unsigned char*>(peer.send.ptr);
            for (std::size_t i = 0; i < peer.send.bytes; ++i) {
                bytes[i] = static_cast<unsigned char>((i * 13 + i / 7 + iteration * 37) % 251);
            }
        }
    }

    static void fill_local_source(Proxy& proxy, uint64_t iteration) {
        auto& buffer = *proxy.local_router_staging_buffer_;
        auto* bytes = static_cast<unsigned char*>(buffer.ptr);
        for (std::size_t i = 0; i < buffer.bytes; ++i)
            bytes[i] = static_cast<unsigned char>((i * 17 + iteration * 29 + proxy.config_.local_gpu_index) % 251);
    }

    static void check_early_local_ready(Proxy& proxy, uint64_t iteration, int shape) {
        const auto& config = proxy.config_;
        const auto before = proxy.router_expert_token_head_state_for_source(
            config.node_rank, config.local_gpu_index);
        if (before.received_token_frontier != 0 ||
            proxy.router_computation_scheduler_state().published_rows != 0)
            throw std::runtime_error("early readiness did not reset across iterations");
        bool rejected = false;
        try { proxy.request_local_router_input_ready(iteration + 1); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("early readiness accepted an unprepared iteration");
        for (const auto invalid : {iteration, std::numeric_limits<uint64_t>::max()}) {
            rejected = false;
            try { proxy.wait_for_local_router_input_ready(invalid); }
            catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("early readiness accepted an unmatched or overflowing wait");
        }

        const auto queued = proxy.nvlink_forward_notifications_enqueued_.load();
        proxy.request_local_router_input_ready(iteration);
        proxy.wait_for_local_router_input_ready(iteration);
        const auto& publication = proxy.cuda_buffers_.router_notification_publication_buffers();
        const auto state = proxy.router_computation_scheduler_state();
        const auto& inputs = proxy.cuda_buffers_.nvlink_receive_buffers();
        const auto cluster_rows = expert_gemm_cluster_rows(config);
        std::size_t expected_batches = 0;
        std::size_t own_slot = inputs.size();
        for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
            const auto& input = inputs[slot];
            const bool own = input.source_node_rank == config.node_rank &&
                input.source_gpu_index == config.local_gpu_index;
            const auto head = proxy.router_expert_token_head_state_for_source(
                input.source_node_rank, input.source_gpu_index);
            const auto expected_frontier = own ?
                proxy.router_routing_.token_indices_for_node(config.node_rank).size() : 0;
            if (head.received_token_frontier != expected_frontier)
                throw std::runtime_error("early readiness advanced the wrong source or x3 range");
            if (!own) continue;
            own_slot = slot;
            const auto& offsets = input.expert_metadata.expert_offsets;
            for (std::size_t expert = 0; expert < state.expert_total_tokens.size(); ++expert) {
                const auto count = static_cast<std::size_t>(offsets[expert + 1] - offsets[expert]);
                const auto batches = count == state.expert_total_tokens[expert] ?
                    (count + cluster_rows - 1) / cluster_rows : count / cluster_rows;
                if (head.expert_token_heads[expert] != count ||
                    state.expert_num_notified_batches[expert] != batches)
                    throw std::runtime_error("early readiness lost expert heads or released a partial cluster");
                expected_batches += batches;
            }
        }
        if (own_slot == inputs.size() || state.published_batches != expected_batches ||
            state.published_rows != expected_batches * publication.n_groups_per_m_cluster ||
            (shape == 0 && expected_batches == 0) || (shape != 0 && expected_batches != 0))
            throw std::runtime_error("early readiness published the wrong gather prefix");
        const auto* table = static_cast<const int32_t*>(publication.host_table.ptr);
        for (std::size_t row = 0; row < state.published_rows; ++row) {
            const auto* ranges = table + row * publication.table_width + 2;
            for (std::size_t slot = 0; slot < inputs.size(); ++slot)
                if (slot != own_slot && ranges[slot * 2] != ranges[slot * 2 + 1])
                    throw std::runtime_error("early gather prefix consumed an unready source");
        }
        if (proxy.local_router_forwarding_completed_iterations_.load() != iteration ||
            proxy.nvlink_forward_notifications_enqueued_.load() != queued)
            throw std::runtime_error("early readiness waited for forwarding or queued a notification");
        // Reentry must not duplicate descriptors or mutate the selected ranges.
        const auto heads = proxy.router_expert_token_head_state_for_source(
            config.node_rank, config.local_gpu_index);
        proxy.request_local_router_input_ready(iteration);
        proxy.wait_for_local_router_input_ready(iteration);
        const auto repeated = proxy.router_expert_token_head_state_for_source(
            config.node_rank, config.local_gpu_index);
        if (proxy.router_computation_scheduler_state().published_rows != state.published_rows ||
            repeated.expert_token_tails != heads.expert_token_tails)
            throw std::runtime_error("early readiness is not idempotent");
    }

    static void verify_early_completion(Proxy& proxy, uint64_t iteration) {
        const auto& config = proxy.config_;
        // With one chunk per batch, independently count direct and compact-copy
        // records. Only the local-self direct records may disappear.
        std::size_t expected_notifications = 0;
        auto count = [&](const auto& chunks, const std::vector<uint8_t>& masks, bool local) {
            if (config.nvlink_forward_notification_scheduling_mode == "complete_round") {
                expected_notifications += std::max(std::size_t{1}, chunks.size()) * config.num_gpus_per_node;
                return; // Fused data/completion, including zero-route and empty-phase markers.
            }
            for (const auto& chunk : chunks) {
                if (!local || !config.router_local_input_early_ready_enabled)
                    ++expected_notifications;
                for (int destination = 0; destination < config.num_gpus_per_node; ++destination) {
                    if (destination == config.local_gpu_index) continue;
                    const auto column = (destination - config.local_gpu_index - 1 +
                        config.num_gpus_per_node) % config.num_gpus_per_node;
                    const auto mask = static_cast<uint8_t>(1U << (7 - column));
                    for (std::size_t token = chunk.start_token;
                         token < chunk.start_token + chunk.num_tokens; ++token) {
                        if (masks.at(token) & mask) { ++expected_notifications; break; }
                    }
                }
            }
        };
        count(proxy.local_router_forwarding_chunks_, proxy.local_router_forwarding_routing_table_, true);
        for (const auto& peer : proxy.peers_)
            count(peer.receive_chunks, peer.forwarding_routing_table, false);
        expected_notifications *= iteration + 1;
        if (proxy.nvlink_forward_notifications_enqueued_.load() != expected_notifications ||
            proxy.nvlink_forward_notifications_sent_.load() != expected_notifications)
            throw std::runtime_error("early readiness changed non-self notifications or left duplicates");

        const auto state = proxy.router_computation_scheduler_state();
        const auto& publication = proxy.cuda_buffers_.router_notification_publication_buffers();
        const auto& inputs = proxy.cuda_buffers_.nvlink_receive_buffers();
        if (state.published_rows != publication.table_rows)
            throw std::runtime_error("early readiness left gather work incomplete");
        std::vector<std::vector<unsigned>> seen;
        for (const auto& input : inputs) seen.emplace_back(input.expert_token_index_count, 0);
        const auto* table = static_cast<const int32_t*>(publication.host_table.ptr);
        for (std::size_t row = 0; row < publication.table_rows;
             row += publication.n_groups_per_m_cluster) {
            const auto* first = table + row * publication.table_width;
            for (std::size_t group = 0; group < publication.n_groups_per_m_cluster; ++group) {
                const auto* repeated = first + group * publication.table_width;
                if (repeated[0] != first[0] || repeated[1] != static_cast<int32_t>(group) * publication.group_size ||
                    !std::equal(first + 2, first + publication.table_width, repeated + 2))
                    throw std::runtime_error("early readiness changed repeated N-group ranges");
            }
            for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
                const auto& metadata = inputs[slot].expert_metadata;
                const auto start = first[2 + 2 * slot], end = first[3 + 2 * slot];
                if (start < metadata.expert_offsets.at(first[0]) ||
                    end > metadata.expert_offsets.at(first[0] + 1) || start > end)
                    throw std::runtime_error("early readiness produced an invalid expert range");
                for (int32_t index = start; index < end; ++index)
                    if (++seen[slot].at(index) != 1)
                        throw std::runtime_error("early readiness duplicated a routed row");
            }
        }
        for (const auto& source : seen)
            if (std::find(source.begin(), source.end(), 0) != source.end())
                throw std::runtime_error("early readiness omitted a routed row");
    }

    static void test_early_local_ready(bool complete_round = false) {
        // Legacy, prepared with ordinary local forwarding, unified epilogue,
        // and atomic/query paths. Dense, partial, no-own-route, and empty inputs
        // run with and without weights; four cases retain the default, followed
        // by legacy/atomic failure and pending-request shutdown cases.
        for (int scenario = 0; scenario < 40; ++scenario) {
            const bool inject_failure = scenario == 36 || scenario == 37;
            const bool shutdown_pending = scenario >= 38;
            const bool lifecycle = scenario >= 36;
            const bool enabled = scenario < 32 || lifecycle;
            const int policy = lifecycle ? (scenario % 2 ? 3 : 0) : scenario % 4;
            const int shape = lifecycle ? 1 : enabled ? (scenario / 4) % 4 : 0;
            const bool check_overlap = enabled && !lifecycle && shape == 0;
            ProxyConfig config;
            config.mock_mode = true;
            config.num_nodes = shape == 2 ? 1 : 2;
            config.num_gpus_per_node = 2;
            config.num_tokens = shape == 0 ? 1025 : shape == 1 ? 17 : 1;
            config.token_dimension = 8;
            config.tokens_per_chunk = 32;
            config.num_iterations = lifecycle ? 1 : 3;
            config.completion_timeout_ms = 10000;
            config.fill_test_data = config.validate_data = false;
            config.router_routing_enabled = true;
            config.router_num_experts = config.num_nodes * 2;
            // Mock RDMA echoes outbound node x3/x4. Selecting every expert
            // makes those mirrored rows agree with reconstructed inbound expert
            // metadata in the two-node cases that assert complete gather work.
            config.router_top_k = shape >= 2 ? 1 : config.router_num_experts;
            config.router_weighted_contributions_enabled = enabled && scenario >= 16;
            config.router_local_input_staging_enabled = true;
            config.router_local_input_early_ready_enabled = enabled;
            config.nvlink_forwarding_enabled = true;
            config.nvlink_forward_completion_notifications_enabled = true;
            if (complete_round) config.nvlink_forward_notification_scheduling_mode = "complete_round";
            config.nvlink_forward_preparation_enabled = policy != 0;
            config.nvlink_forward_local_staging_prepared_enabled = policy >= 2;
            config.nvlink_forward_submit_epilogue_enabled = policy >= 2;
            config.nvlink_forward_prepared_atomic_ring_enabled = policy == 3;
            config.nvlink_forward_completion_mode = policy == 3 ? "stream_query" : "stream_sync";
            config.nvlink_forward_prepared_batch_chunks = 1;
            config.nvlink_forward_chunk_tokens = config.tokens_per_chunk;
            config.nvlink_forward_min_threshold_chunks = 1;
            config.nvlink_forward_max_threshold_chunks = 1;
            config.nvlink_forward_prepared_queue_depth = 2;
            config.nvlink_forward_notification_queue_depth = 1;
            config.nvlink_forward_synchronize_batches = true;
            config.nvlink_forward_synchronize_iteration = false;
            config.nvlink_forward_local_batch_sync_enabled = true;
            config.local_iteration_sync_enabled = true;
            config.sequential_peer_transfers = true;
            config.local_forwarding_rdma_overlap_enabled = scenario % 2 == 0;
            config.expert_gemm_dimension = 768;
            config.expert_gemm_n_tile = 128;
            config.expert_gemm_max_swizzle_size = 2; // Three repeated N groups.
            config.expert_gemm_device_capacity = scenario % 2 ? 10 : 9;
            config.expert_gemm_cluster_m = scenario % 2 ? 2 : 1;
            config.nvlink_forward_notification_flush_per_entry_enabled = scenario % 2;
            const auto nonce = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
            config.local_iteration_sync_run_id = "test_early_local_" + nonce;
            config.nvlink_forward_exchange_dir = "/tmp/test_early_local_" + nonce;
            if (config.num_nodes > 1) config.peers.push_back(PeerAddress{1, "mock-peer", 18515});
            if (shape >= 2) {
                // Use one seed on all actual proxies: mock expert exchange
                // reconstructs other sources using the receiver's seed. Shape
                // two selects only GPU1 locally; shape three has no local x3.
                for (uint64_t seed = 1;; ++seed) {
                    config.router_seed = seed;
                    RouterRouting routing(config);
                    routing.initialize();
                    const bool matches = shape == 2 ?
                        routing.expert_metadata_for_gpu(0, 0).expert_token_indices.empty() :
                        routing.token_indices_for_node(0).empty();
                    if (matches) break;
                    if (seed == 10000) throw std::runtime_error("could not find empty-route fixture seed");
                }
            }
            std::cerr << "early local readiness scenario=" << scenario << " policy=" << policy
                      << " shape=" << shape << " enabled=" << enabled << '\n';
            std::array<EarlyReadyReceiverGate, 2> gates;
            if (check_overlap) gates[0].pause_call = 2; // Production-only iteration one.
            if (inject_failure) for (auto& gate : gates) gate.fail_call = 1;
            if (shutdown_pending) for (auto& gate : gates) gate.pause_call = 1;
            std::array<std::unique_ptr<Proxy>, 2> proxies;
            for (int gpu = 0; gpu < 2; ++gpu) {
                auto cfg = config;
                cfg.local_gpu_index = cfg.cuda_device_id = gpu;
                validate_config(cfg);
                proxies[gpu] = std::make_unique<Proxy>(cfg);
                proxies[gpu]->local_router_input_early_ready_test_hook_ = [&gates, gpu] { gates[gpu](); };
            }
            std::array<std::exception_ptr, 2> errors;
            std::atomic<unsigned> phase{0};
            std::atomic<bool> failed{false};
            auto barrier = [&](unsigned target) {
                phase.fetch_add(1);
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                while (phase.load() < target) {
                    if (failed.load() || std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("early local readiness test barrier cancelled");
                    std::this_thread::yield();
                }
            };
            auto run = [&](int gpu) {
                try {
                    auto& proxy = *proxies[gpu];
                    proxy.initialize();
                    if (lifecycle) {
                        proxy.prepare_iteration_step(0);
                        proxy.request_local_router_input_ready(0);
                        if (inject_failure) {
                            bool observed_failure = false;
                            try { proxy.wait_for_local_router_input_ready(0); }
                            catch (const std::runtime_error& error) {
                                if (std::string(error.what()).find("injected early-ready receiver failure") == std::string::npos)
                                    throw;
                                observed_failure = true;
                            }
                            if (!observed_failure || proxy.local_router_input_early_ready_completed_.load() != 0)
                                throw std::runtime_error("receiver failure acknowledged unfinished local input");
                            gates[gpu].verify(proxy.nvlink_forward_notification_thread_.get_id(), 1);
                        }
                        return;
                    }
                    for (uint64_t iteration = 0; iteration < config.num_iterations; ++iteration) {
                        fill_source(proxy, iteration);
                        proxy.prepare_iteration_step(iteration);
                        // Source staging is immutable throughout this fixture.
                        if (iteration == 0) fill_local_source(proxy, 0);
                        if (enabled && iteration != 1) check_early_local_ready(proxy, iteration, shape);
                        barrier(static_cast<unsigned>(iteration * 6 + 2));
                        // Iteration one exercises the production activation call
                        // with no manual early publication from the test.
                        proxy.run_iteration_step(iteration);
                        // Ordinary legacy forwarding has never promised queue
                        // draining at run return; explicitly retire its records
                        // before inspecting receiver state or reusing inputs.
                        if (!config.nvlink_forward_preparation_enabled)
                            proxy.drain_forwarding_iteration_notifications(iteration);
                        if (enabled && proxy.local_router_input_early_ready_completed_.load(std::memory_order_acquire) != iteration + 1)
                            throw std::runtime_error("iteration retired before receiver acknowledged local input");
                        gates[gpu].verify(proxy.nvlink_forward_notification_thread_.get_id(), enabled ? iteration + 1 : 0);
                        barrier(static_cast<unsigned>(iteration * 6 + 4));
                        verify(proxy, *proxies[1 - gpu], iteration);
                        verify_early_completion(proxy, iteration);
                        barrier(static_cast<unsigned>(iteration * 6 + 6));
                    }
                    proxy.finish_run();
                } catch (...) { errors[gpu] = std::current_exception(); failed.store(true); }
            };
            std::thread first(run, 0), second(run, 1);
            std::exception_ptr observer_error;
            if (check_overlap) {
                try {
                    gates[0].wait_paused();
                    auto& proxy = *proxies[0];
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    while (true) {
                        bool copied = false;
                        {
                            std::lock_guard<std::mutex> lock(proxy.forwarding_mutex_);
                            copied = proxy.forwarding_iteration_stats_.size() > 1 &&
                                proxy.forwarding_iteration_stats_[1].batch_count > 0;
                        }
                        if (copied) break;
                        if (failed.load() || std::chrono::steady_clock::now() >= deadline)
                            throw std::runtime_error("forwarding waited for the receiver's early publication");
                        std::this_thread::yield();
                    }
                    if (proxy.router_expert_token_head_state_for_source(0, 0).received_token_frontier != 0 ||
                        proxy.local_router_input_early_ready_completed_.load(std::memory_order_acquire) != 1)
                        throw std::runtime_error("held receiver published or acknowledged local input");
                } catch (...) { observer_error = std::current_exception(); failed.store(true); }
                gates[0].release();
            }
            first.join(); second.join();
            if (shutdown_pending) {
                try {
                    for (int gpu = 0; gpu < 2; ++gpu) {
                        gates[gpu].wait_paused();
                        gates[gpu].verify(proxies[gpu]->nvlink_forward_notification_thread_.get_id(), 1);
                    }
                } catch (...) { observer_error = std::current_exception(); }
            }
            std::atomic<unsigned> stopped{0};
            auto stop = [&](int gpu) {
                try { proxies[gpu]->shutdown(); }
                catch (const std::runtime_error& error) {
                    const bool expected_cancel = complete_round && shutdown_pending &&
                        std::string(error.what()).find("incomplete forwarding rounds") != std::string::npos;
                    if (!expected_cancel && (!inject_failure || std::string(error.what()).find("injected early-ready receiver failure") == std::string::npos)) {
                        if (!errors[gpu]) errors[gpu] = std::current_exception();
                    }
                }
                catch (...) { if (!errors[gpu]) errors[gpu] = std::current_exception(); }
                stopped.fetch_add(1);
            };
            std::thread stop0(stop, 0), stop1(stop, 1);
            if (shutdown_pending) {
                try {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    while (!proxies[0]->nvlink_forward_notification_receiver_stop_.load() ||
                           !proxies[1]->nvlink_forward_notification_receiver_stop_.load()) {
                        if (std::chrono::steady_clock::now() >= deadline)
                            throw std::runtime_error("shutdown did not reach the pending receiver");
                        std::this_thread::yield();
                    }
                    if (stopped.load() != 0)
                        throw std::runtime_error("shutdown returned while the receiver callback was active");
                } catch (...) { if (!observer_error) observer_error = std::current_exception(); }
            }
            for (auto& gate : gates) gate.release();
            stop0.join(); stop1.join();
            // An initialization failure can prevent the observer from ever
            // seeing its gate. Retain that cause alongside the gate timeout.
            if (observer_error) {
                for (int gpu = 0; gpu < 2; ++gpu) {
                    if (!errors[gpu]) continue;
                    try { std::rethrow_exception(errors[gpu]); }
                    catch (const std::exception& error) {
                        std::cerr << "early local readiness scenario=" << scenario
                                  << " gpu=" << gpu << " error=" << error.what() << '\n';
                    }
                    catch (...) { std::cerr << "early local readiness nonstandard worker exception\n"; }
                }
            }
            if (observer_error) std::rethrow_exception(observer_error);
            for (auto error : errors) if (error) std::rethrow_exception(error);
        }
    }

    static void verify(Proxy& source, Proxy& receiver, uint64_t iteration) {
        const auto& config = source.config_;
        if (!config.nvlink_forward_completion_notifications_enabled &&
            (source.nvlink_forward_notification_dispatch_ ||
             source.nvlink_forward_notification_header_ ||
             source.nvlink_forward_notification_dispatch_thread_.joinable() ||
             source.nvlink_forward_notification_thread_.joinable() ||
             source.nvlink_forward_notifications_enqueued_.load() != 0)) {
            throw std::runtime_error("disabled notifications still allocated or published");
        }
        const auto token_bytes = config.token_dimension * dtype_size(config.dtype);
        std::size_t remote_batches = 0;
        auto check_input = [&](int node, const void* input, const std::vector<uint8_t>& masks) {
            const auto column = (receiver.config_.local_gpu_index - config.local_gpu_index - 1 +
                config.num_gpus_per_node) % config.num_gpus_per_node;
            const auto mask = static_cast<uint8_t>(1U << (7 - column));
            const auto& output = receiver.cuda_buffers_.nvlink_receive_buffer_for_source(
                node, config.local_gpu_index).recv;
            std::size_t compacted = 0;
            for (std::size_t token = 0; token < masks.size(); ++token) {
                if ((masks[token] & mask) == 0) continue;
                if (std::memcmp(static_cast<const char*>(input) + token * token_bytes,
                                static_cast<const char*>(output.ptr) + compacted * token_bytes,
                                token_bytes) != 0) {
                    throw std::runtime_error("prepared forwarding compacted payload mismatch");
                }
                ++compacted;
            }
            if (!config.nvlink_forward_completion_notifications_enabled) return;
            const auto head = receiver.router_expert_token_head_state_for_source(node, config.local_gpu_index);
            if (head.received_token_frontier != compacted || (compacted && head.iteration != iteration)) {
                throw std::runtime_error("prepared forwarding returned before notifications drained: source_node=" +
                    std::to_string(node) + " source_gpu=" + std::to_string(config.local_gpu_index) +
                    " destination_gpu=" + std::to_string(receiver.config_.local_gpu_index) +
                    " expected_frontier=" + std::to_string(compacted) +
                    " actual_frontier=" + std::to_string(head.received_token_frontier) +
                    " expected_iteration=" + std::to_string(iteration) +
                    " actual_iteration=" + std::to_string(head.iteration));
            }
            const auto direct = source.router_expert_token_head_state_for_source(node, config.local_gpu_index);
            if (direct.received_token_frontier != masks.size() ||
                (!masks.empty() && direct.iteration != iteration)) {
                throw std::runtime_error("direct input notification range mismatch");
            }
        };
        for (std::size_t p = 0; p < source.peers_.size(); ++p) {
            const auto& peer = source.peers_[p];
            const auto chunks = peer.receive_chunks.size();
            remote_batches += (chunks + config.nvlink_forward_prepared_batch_chunks - 1) /
                config.nvlink_forward_prepared_batch_chunks;
            check_input(peer.peer_rank, source.cuda_buffers_.peer_buffers()[p].recv.ptr,
                        peer.forwarding_routing_table);
            if (source.forwarding_next_chunk_by_peer_[p] != (iteration + 1) * chunks) {
                throw std::runtime_error("completion cursor mismatch");
            }
        }
        std::size_t local_batches = 0;
        if (config.router_local_input_staging_enabled) {
            if (source.local_router_forwarding_completed_iterations_.load() != iteration + 1)
                throw std::runtime_error("local forwarding returned before epilogue completion");
            const auto chunks = source.local_router_forwarding_chunks_.size();
            local_batches = (chunks + config.nvlink_forward_prepared_batch_chunks - 1) /
                config.nvlink_forward_prepared_batch_chunks;
            check_input(config.node_rank, source.local_router_staging_buffer_->ptr,
                        source.local_router_forwarding_routing_table_);
        }
        std::lock_guard<std::mutex> lock(source.forwarding_mutex_);
        if (source.forwarding_iteration_stats_.at(iteration).batch_count != remote_batches + local_batches) {
            throw std::runtime_error("barrier changed a fixed batch size");
        }
    }

    static void fail(Proxy& proxy) { proxy.set_forwarding_error("injected prepared failure"); }
    static void invalidate_destination(Proxy& proxy) {
        for (auto& destination : proxy.forwarding_destinations_) {
            for (auto& buffer : destination.source_buffers) buffer.bytes = 0;
        }
    }
};
} // namespace rdma_proxy

int main(int argc, char** argv) {
    using namespace rdma_proxy;
    Logger::instance().set_level(LogLevel::kError);
    const std::string mode_arg = argc == 2 ? argv[1] : "";
    if (mode_arg == "--early-local-ready" || mode_arg == "--complete-round") {
        PreparedForwardingTestAccess::test_early_local_ready(mode_arg == "--complete-round");
        std::cout << "test_early_local_ready passed\n";
        return 0;
    }
    const bool complete_round_forwarding = mode_arg == "--complete-round-forwarding";
    const bool dual = mode_arg == "--dual-device";
    const bool unified = dual || mode_arg == "--unified-local" || mode_arg == "--unified-local-atomic" ||
        mode_arg == "--unified-local-query";
    const bool ping_pong2_only = argc == 2 && std::string(argv[1]) == "--ping-pong2-only";
    const bool query_atomic = argc == 2 && std::string(argv[1]) == "--stream-query-atomic";
    const bool query_mode = mode_arg == "--unified-local-query" || query_atomic || (argc == 2 && std::string(argv[1]) == "--stream-query");
    const bool atomic_ring_depth4 = argc == 2 && std::string(argv[1]) == "--atomic-ring-depth4";
    const bool atomic_ring_only = dual || mode_arg == "--unified-local-atomic" || query_atomic || atomic_ring_depth4 ||
        (argc == 2 && std::string(argv[1]) == "--atomic-ring-only");
    const bool submit_epilogue_only = argc == 2 && std::string(argv[1]) == "--submit-epilogue-only";
    if (argc != 1 && !complete_round_forwarding && !unified && !ping_pong2_only &&
        !submit_epilogue_only && !atomic_ring_only && !query_mode)
        throw std::runtime_error("unknown test argument");
    PreparedForwardingTestAccess::test_barrier_generations(ping_pong2_only);
    if (complete_round_forwarding) PreparedForwardingTestAccess::test_barrier_generations(true);
    if (ping_pong2_only || complete_round_forwarding)
        PreparedForwardingTestAccess::test_ping_pong2_dispatch_order();
    // Two-entry ring is exercised over many batches/iterations; one-entry
    // notification rings force dispatch backpressure. n=1 includes empty-copy
    // batches; n=3 gives unequal tails; large n gives a single short batch.
    // Exercise both notification modes with and without the batch barrier.
    // Complete-round mode runs the same shapes, no-early-local behavior and
    // lifecycle failures through six forwarding policies. Keep both barrier
    // settings, but notifications must stay enabled for explicit round markers.
    // Policies: prepared coordinator, ping-pong2, submit/epilogue, atomic ring,
    // stream-query mutex, and stream-query atomic. Other arguments retain their
    // original per-notification and notification-disabled baseline matrices.
    const int scenario_end = complete_round_forwarding ? 6 * 18 : ping_pong2_only ? 72 : 36;
    for (int scenario = ping_pong2_only ? 36 : 0; scenario < scenario_end; ++scenario) {
        const int mode = scenario % 9;
        const int policy = complete_round_forwarding ? scenario / 18 : -1;
        const bool local_batch_sync = (scenario / 9) % 2 == 0;
        const bool notifications = complete_round_forwarding || scenario % 36 < 18;
        const bool ping_pong2 = complete_round_forwarding ? policy == 1 : scenario >= 36;
        const bool scenario_atomic = atomic_ring_only ||
            (complete_round_forwarding && (policy == 3 || policy == 5));
        const bool scenario_query = query_mode || (complete_round_forwarding && policy >= 4);
        const bool scenario_submit = unified || submit_epilogue_only || scenario_atomic ||
            scenario_query || (complete_round_forwarding && policy == 2);
        std::cerr << "prepared forwarding test mode=" << mode
                  << " scheduling=" << (complete_round_forwarding ? "complete_round" : "per_notification")
                  << " local_batch_sync=" << local_batch_sync
                  << " ping_pong2=" << ping_pong2
                  << " submit_epilogue=" << scenario_submit
                  << " atomic_ring=" << scenario_atomic
                  << " stream_query=" << scenario_query
                  << " notifications=" << notifications << '\n';
        ProxyConfig config;
        config.node_rank = 0;
        config.num_nodes = unified && mode == 5 ? 1 : mode == 4 ? 3 : 2;
        config.num_gpus_per_node = 2;
        config.num_tokens = mode == 3 ? 1 : 137;
        config.token_dimension = 8;
        config.tokens_per_chunk = mode == 0 ? 1 : 4;
        config.num_qps_per_peer = 2;
        if (dual) {
            config.rdma_device_name = "mock0"; config.rdma_device_name2 = "mock1";
            config.num_qps_per_peer = 4;
        }
        config.num_iterations = 3;
        config.completion_timeout_ms = 10000;
        config.mock_mode = true;
        config.fill_test_data = false;
        config.validate_data = false;
        config.router_routing_enabled = true;
        config.router_num_experts = config.num_nodes * 2;
        config.router_top_k = 1;
        config.nvlink_forwarding_enabled = true;
        config.nvlink_forward_preparation_enabled = true;
        config.nvlink_forward_ping_pong2_enabled = ping_pong2;
        config.nvlink_forward_local_staging_prepared_enabled = unified;
        config.nvlink_forward_submit_epilogue_enabled = scenario_submit;
        config.nvlink_forward_completion_mode = scenario_query ? "stream_query" : "stream_sync";
        config.nvlink_forward_prepared_atomic_ring_enabled = scenario_atomic;
        config.nvlink_forward_prepared_batch_chunks = mode == 0 ? 1 : mode == 2 ? 1000 : 3;
        config.nvlink_forward_prepared_queue_depth = atomic_ring_depth4 ? 4 : 2;
        config.nvlink_forward_synchronize_batches = true;
        config.nvlink_forward_synchronize_iteration = false;
        config.nvlink_forward_local_batch_sync_enabled = local_batch_sync;
        config.nvlink_forward_completion_notifications_enabled = notifications;
        if (complete_round_forwarding)
            config.nvlink_forward_notification_scheduling_mode = "complete_round";
        config.nvlink_forward_notification_queue_depth = 1;
        config.local_iteration_sync_enabled = true;
        config.router_local_input_staging_enabled = unified || mode < 5;
        config.sequential_peer_transfers = true;
        config.local_forwarding_rdma_overlap_enabled = !unified || mode % 2 == 0;
        // Deliberately conflicting legacy settings are ignored in this mode.
        config.nvlink_forward_min_threshold_chunks = 32;
        config.nvlink_forward_max_threshold_chunks = 64;
        config.nvlink_forward_threshold_tokens = 1024;
        config.nvlink_forward_threshold_chunks = 8;
        const auto nonce = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        config.local_iteration_sync_run_id = "test_prepared_" + nonce;
        config.nvlink_forward_exchange_dir = "/tmp/test_prepared_" + nonce;
        for (int node = 1; node < config.num_nodes; ++node) {
            config.peers.push_back(PeerAddress{node, "mock-peer", 18515});
        }
        std::array<std::unique_ptr<Proxy>, 2> proxies;
        for (int gpu = 0; gpu < 2; ++gpu) {
            auto cfg = config;
            cfg.local_gpu_index = gpu;
            cfg.cuda_device_id = gpu;
            cfg.router_seed = gpu == 0 ? 1 : 128;
            // Ensure one zero-remote-input proxy and one nonempty proxy.
            if (mode == 3) {
                for (uint64_t seed = 1;; ++seed) {
                    cfg.router_seed = seed;
                    RouterRouting routing(cfg);
                    routing.initialize();
                    if ((routing.token_indices_for_node(1).empty()) == (gpu == 0)) break;
                }
            }
            validate_config(cfg);
            proxies[gpu] = std::make_unique<Proxy>(cfg);
        }
        std::array<std::exception_ptr, 2> errors;
        std::atomic<unsigned> checked{0};
        std::atomic<bool> failed{false};
        auto run = [&](int gpu) {
            try {
                auto& proxy = *proxies[gpu];
                proxy.initialize();
                if (mode == 6) return; // shutdown while both pipeline threads wait
                if (mode == 7) { PreparedForwardingTestAccess::fail(proxy); return; }
                if (mode == 8) PreparedForwardingTestAccess::invalidate_destination(proxy);
                int32_t* trace_flag = nullptr;
                if (notifications && mode < 6)
                    trace_flag = reinterpret_cast<int32_t*>(proxy.configure_down_ready_trace(3));
                for (uint64_t i = 0; i < config.num_iterations; ++i) {
                    PreparedForwardingTestAccess::fill_source(proxy, i);
                    if (unified || trace_flag) {
                        proxy.prepare_iteration_step(i);
                    }
                    if (unified) {
                        PreparedForwardingTestAccess::fill_local_source(proxy, i);
                    }
                    if (trace_flag) {
                        if (__atomic_load_n(trace_flag, __ATOMIC_ACQUIRE) != 0)
                            throw std::runtime_error("stale down-ready flag across iterations");
                        __atomic_store_n(trace_flag, 1, __ATOMIC_RELEASE);
                    }
                    proxy.run_iteration_step(i);
                    PreparedForwardingTestAccess::verify(proxy, *proxies[1 - gpu], i);
                    if (trace_flag) {
                        __atomic_store_n(trace_flag, 3, __ATOMIC_RELEASE);
                        const auto trace = proxy.read_down_ready_trace(i);
                        if (trace.empty() || trace.back().first != 3)
                            throw std::runtime_error("receiver missed final down-ready prefix");
                    }
                    checked.fetch_add(1);
                    while (checked.load() < (i + 1) * 2 && !failed.load()) std::this_thread::yield();
                    if (failed.load()) return;
                }
                proxy.finish_run();
            } catch (const std::exception& error) {
                std::cerr << "gpu=" << gpu << " error=" << error.what() << '\n';
                errors[gpu] = std::current_exception();
                failed.store(true);
            }
        };
        std::thread first(run, 0), second(run, 1);
        first.join(); second.join();
        // Shutdown concurrently: each receiver drains until all senders stop.
        auto shutdown = [&](int gpu) {
            try { proxies[gpu]->shutdown(); }
            catch (...) { if (!errors[gpu]) errors[gpu] = std::current_exception(); }
        };
        const auto stop_start = std::chrono::steady_clock::now();
        std::thread stop0(shutdown, 0), stop1(shutdown, 1);
        stop0.join(); stop1.join();
        if (mode >= 6 && std::chrono::steady_clock::now() - stop_start > std::chrono::seconds(5)) {
            throw std::runtime_error("prepared shutdown waited for the completion timeout");
        }
        for (auto error : errors) {
            if (mode >= 7) {
                if (!error) throw std::runtime_error("prepared failure was not propagated");
                try { std::rethrow_exception(error); }
                catch (const std::runtime_error& failure) {
                    const std::string message = failure.what();
                    const char* expected = mode == 7 ? "injected prepared failure" : "copy exceeds buffer bounds";
                    if (message.find(expected) == std::string::npos) throw;
                }
            } else if (error) std::rethrow_exception(error);
        }
    }

    // Final argument arrays are reusable and submission reads exactly them.
    CudaPreparedForwardBatch args;
    args.reserve(2);
    std::array<int, 4> input{1, 2, 3, 4}, output{};
    args.append(output.data(), input.data() + 1, sizeof(int));
    args.append(output.data() + 1, input.data() + 3, sizeof(int));
    const auto* storage = args.dsts.data();
    launch_cuda_prepared_forward_batch_async(args, nullptr, true);
    if (output[0] != 2 || output[1] != 4) throw std::runtime_error("prepared argument mismatch");
    args.clear();
    args.append(output.data(), input.data(), sizeof(int));
    if (args.dsts.data() != storage) throw std::runtime_error("prepared storage was not reused");
    launch_cuda_prepared_forward_batch_async(args, nullptr, true);
    if (output[0] != 1) throw std::runtime_error("reused argument mismatch");
    std::cout << "test_prepared_forwarding passed\n";
}
