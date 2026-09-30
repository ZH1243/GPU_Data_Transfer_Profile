#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace rdma_proxy {
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
                throw std::runtime_error("prepared forwarding returned before notifications drained");
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
    const bool ping_pong2_only = argc == 2 && std::string(argv[1]) == "--ping-pong2-only";
    const bool query_atomic = argc == 2 && std::string(argv[1]) == "--stream-query-atomic";
    const bool query_mode = query_atomic || (argc == 2 && std::string(argv[1]) == "--stream-query");
    const bool atomic_ring_depth4 = argc == 2 && std::string(argv[1]) == "--atomic-ring-depth4";
    const bool atomic_ring_only = query_atomic || atomic_ring_depth4 ||
        (argc == 2 && std::string(argv[1]) == "--atomic-ring-only");
    const bool submit_epilogue_only = argc == 2 && std::string(argv[1]) == "--submit-epilogue-only";
    if (argc != 1 && !ping_pong2_only && !submit_epilogue_only && !atomic_ring_only && !query_mode) throw std::runtime_error("unknown test argument");
    PreparedForwardingTestAccess::test_barrier_generations(ping_pong2_only);
    if (ping_pong2_only) PreparedForwardingTestAccess::test_ping_pong2_dispatch_order();
    // Two-entry ring is exercised over many batches/iterations; one-entry
    // notification rings force dispatch backpressure. n=1 includes empty-copy
    // batches; n=3 gives unequal tails; large n gives a single short batch.
    // Exercise both notification modes with and without the batch barrier.
    for (int scenario = ping_pong2_only ? 36 : 0; scenario < (ping_pong2_only ? 72 : 36); ++scenario) {
        const int mode = scenario % 9;
        const bool local_batch_sync = (scenario / 9) % 2 == 0;
        const bool notifications = scenario % 36 < 18;
        const bool ping_pong2 = scenario >= 36;
        std::cerr << "prepared forwarding test mode=" << mode
                  << " local_batch_sync=" << local_batch_sync
                  << " ping_pong2=" << ping_pong2
                  << " submit_epilogue=" << submit_epilogue_only
                  << " atomic_ring=" << atomic_ring_only
                  << " notifications=" << notifications << '\n';
        ProxyConfig config;
        config.node_rank = 0;
        config.num_nodes = mode == 4 ? 3 : 2;
        config.num_gpus_per_node = 2;
        config.num_tokens = mode == 3 ? 1 : 137;
        config.token_dimension = 8;
        config.tokens_per_chunk = mode == 0 ? 1 : 4;
        config.num_qps_per_peer = 2;
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
        config.nvlink_forward_submit_epilogue_enabled = submit_epilogue_only || atomic_ring_only || query_mode;
        config.nvlink_forward_completion_mode = query_mode ? "stream_query" : "stream_sync";
        config.nvlink_forward_prepared_atomic_ring_enabled = atomic_ring_only;
        config.nvlink_forward_prepared_batch_chunks = mode == 0 ? 1 : mode == 2 ? 1000 : 3;
        config.nvlink_forward_prepared_queue_depth = atomic_ring_depth4 ? 4 : 2;
        config.nvlink_forward_synchronize_batches = true;
        config.nvlink_forward_synchronize_iteration = false;
        config.nvlink_forward_local_batch_sync_enabled = local_batch_sync;
        config.nvlink_forward_completion_notifications_enabled = notifications;
        config.nvlink_forward_notification_queue_depth = 1;
        config.local_iteration_sync_enabled = true;
        config.router_local_input_staging_enabled = mode < 5;
        config.sequential_peer_transfers = true;
        config.local_forwarding_rdma_overlap_enabled = true;
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
                for (uint64_t i = 0; i < config.num_iterations; ++i) {
                    PreparedForwardingTestAccess::fill_source(proxy, i);
                    proxy.run_iteration_step(i);
                    PreparedForwardingTestAccess::verify(proxy, *proxies[1 - gpu], i);
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
