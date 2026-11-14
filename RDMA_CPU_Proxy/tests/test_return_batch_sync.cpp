#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function action, const std::string& text) {
    try { action(); }
    catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(text) != std::string::npos) return;
        throw std::runtime_error("unexpected return barrier error: " + std::string(error.what()));
    }
    throw std::runtime_error("expected return barrier rejection: " + text);
}

rdma_proxy::ProxyConfig config(const std::string& suffix) {
    rdma_proxy::ProxyConfig cfg;
    cfg.mock_mode = true;
    cfg.num_gpus_per_node = 3;
    cfg.num_nodes = 1;
    cfg.num_iterations = 2;
    cfg.router_routing_enabled = true;
    cfg.router_local_input_staging_enabled = true;
    cfg.nvlink_forward_completion_notifications_enabled = true;
    cfg.nvlink_return_completion_notifications_enabled = true;
    cfg.nvlink_return_local_batch_sync_enabled = true;
    cfg.nvlink_forward_notification_scheduling_mode = "complete_round";
    cfg.nvlink_forward_preparation_enabled = true;
    cfg.nvlink_forward_submit_epilogue_enabled = true;
    cfg.nvlink_forward_prepared_atomic_ring_enabled = true;
    cfg.nvlink_forward_notification_queue_depth = 3;
    cfg.completion_timeout_ms = 5000;
    cfg.local_iteration_sync_run_id = "test_return_batch_" + suffix + "_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    return cfg;
}
}  // namespace

namespace rdma_proxy {
// Use the existing friend seam in a separate executable. Real POSIX queues,
// notification threads, down-readiness polling, and mock CPU copies exercise
// the same integration as production without requiring CUDA or an RDMA NIC.
struct ReturnNotificationsTestAccess {
    static constexpr int world = 3, groups = 3, width = 9, tile = 2, group_tiles = 2;
    static constexpr int padded_width = 10, capacity = 24;
    using Buffer = std::array<unsigned char, capacity>;

    struct Fixture {
        ProxyConfig cfg;
        std::array<std::array<Buffer, world>, world> src{}, dst{};
        std::array<int64_t, world> rows{2, 1, 0};
        std::array<std::unique_ptr<Proxy>, world> proxies;
        std::array<int32_t*, world> flags{};
        bool stopped = false;

        explicit Fixture(bool atomic_ring) : cfg(config(atomic_ring ? "atomic" : "mutex")) {
            cfg.nvlink_forward_prepared_atomic_ring_enabled = atomic_ring;
        }
        ~Fixture() { try { stop(); } catch (...) {} }

        void start() {
            for (int gpu = 0; gpu < world; ++gpu) {
                cfg.local_gpu_index = gpu;
                proxies[gpu] = std::make_unique<Proxy>(cfg);
                auto& proxy = *proxies[gpu];
                for (int peer = 0; peer < world; ++peer) {
                    if (peer == gpu) continue;
                    Proxy::ForwardDestinationState destination;
                    destination.gpu_index = peer;
                    proxy.forwarding_destinations_.push_back(destination);
                }
                proxy.initialize_nvlink_forward_notifications();
                proxy.initialize_prepared_forwarding();
                proxy.initialize_nvlink_forward_notification_dispatch();
                proxy.initialize_forwarding_rounds();
            }
            for (auto& proxy : proxies) {
                proxy->prepare_forwarding_notification_destinations();
                proxy->initialized_ = true;
                proxy->nvlink_forward_notification_dispatch_thread_ =
                    std::thread(&Proxy::nvlink_forward_notification_dispatch_loop, proxy.get());
                proxy->nvlink_forward_notification_thread_ =
                    std::thread(&Proxy::nvlink_forward_notification_loop, proxy.get());
                proxy->forwarding_thread_ = std::thread(&Proxy::forwarding_loop, proxy.get(), 0);
            }
            for (int gpu = 0; gpu < world; ++gpu) {
                auto& proxy = *proxies[gpu];
                flags[gpu] = reinterpret_cast<int32_t*>(proxy.configure_down_ready_trace(groups));
                std::vector<RdmaProxyReturnCopy> copies;
                for (int peer = 0; peer < world; ++peer) {
                    if (peer == gpu) continue;
                    copies.push_back({reinterpret_cast<uintptr_t>(src[gpu][peer].data()),
                        reinterpret_cast<uintptr_t>(dst[peer][gpu].data()), capacity, 0, peer});
                }
                proxy.configure_return_copies(copies.data(), copies.size(), &rows[gpu], 1,
                                              width, tile, group_tiles, 1, true, 0);
            }
        }

        void stop() {
            if (stopped) return;
            stopped = true;
            std::array<std::thread, world> threads;
            std::array<std::exception_ptr, world> errors{};
            for (int gpu = 0; gpu < world; ++gpu)
                if (proxies[gpu]) threads[gpu] = std::thread([&, gpu] {
                    try { proxies[gpu]->shutdown(); }
                    catch (...) { errors[gpu] = std::current_exception(); }
                });
            for (auto& thread : threads) if (thread.joinable()) thread.join();
            for (auto error : errors) if (error) std::rethrow_exception(error);
        }

        template <class Condition>
        void wait(Condition ready) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!ready()) {
                for (auto& proxy : proxies) proxy->check_forwarding_error();
                require(std::chrono::steady_clock::now() < deadline, "return batch fixture timed out");
                std::this_thread::yield();
            }
        }

        void check_copies(int completed_groups) const {
            const auto columns = std::min(completed_groups * tile * group_tiles, padded_width);
            for (int dest = 0; dest < world; ++dest)
                for (int producer = 0; producer < world; ++producer) {
                    if (dest == producer) continue;
                    for (int byte = 0; byte < capacity; ++byte)
                        require(dst[dest][producer][byte] ==
                                    (byte < rows[producer] * columns ? src[producer][dest][byte] : 255),
                                "return batch copied early, missed data, or overwrote padding/capacity");
                }
        }

        void check_sent(uint64_t expected) const {
            for (auto& proxy : proxies)
                require(proxy->return_notifications_sent_.load(std::memory_order_acquire) == expected,
                        "return batch completion escaped the start barrier");
        }

        void exercise() {
            start();
            for (uint64_t iteration = 0; iteration < 2; ++iteration) {
                // Reverse which producer is empty in the second iteration.
                // The empty one remains an expected participant in every group.
                rows = iteration ? std::array<int64_t, world>{0, 2, 1} :
                                   std::array<int64_t, world>{2, 1, 0};
                const int delayed = iteration ? 0 : 2;
                for (int gpu = 0; gpu < world; ++gpu) {
                    for (int peer = 0; peer < world; ++peer) {
                        dst[gpu][peer].fill(255);
                        for (int byte = 0; byte < capacity; ++byte)
                            src[gpu][peer][byte] = static_cast<unsigned char>(1 + gpu * 30 + byte + iteration * 100);
                    }
                    auto& proxy = *proxies[gpu];
                    proxy.return_group_progress_.reset(iteration);
                    proxy.down_ready_trace_.reset(iteration);
                    proxy.forwarding_round_requested_.store(iteration + 1, std::memory_order_release);
                    proxy.arm_return_copies(iteration);
                }
                // Fast producers observe the final prefix immediately. They
                // must still wait at group zero for the delayed empty producer.
                for (int gpu = 0; gpu < world; ++gpu)
                    if (gpu != delayed) __atomic_store_n(flags[gpu], groups, __ATOMIC_RELEASE);

                // Forwarding completions use the same receiver that polls the
                // return gate. They must keep draining while that gate waits.
                for (auto& proxy : proxies) proxy->enqueue_empty_forwarding_phases(iteration);
                wait([&] {
                    return std::all_of(proxies.begin(), proxies.end(), [&](const auto& proxy) {
                        return proxy->nvlink_forward_notifications_received_.load() == (iteration + 1) * world;
                    });
                });
                check_sent(iteration * groups);
                check_copies(0);

                __atomic_store_n(flags[delayed], 1, __ATOMIC_RELEASE);
                wait([&] {
                    return std::all_of(proxies.begin(), proxies.end(), [&](const auto& proxy) {
                        return proxy->return_ready_prefix(iteration) == 1 &&
                            proxy->return_notifications_sent_.load(std::memory_order_acquire) >= iteration * groups + 1;
                    });
                });
                check_sent(iteration * groups + 1);
                check_copies(1);

                // A second jump releases every intermediate group, including
                // the final partial group. Publication must follow its copies.
                __atomic_store_n(flags[delayed], groups, __ATOMIC_RELEASE);
                for (auto& proxy : proxies) proxy->read_down_ready_trace(iteration);
                check_sent((iteration + 1) * groups);
                check_copies(groups);
                for (auto& proxy : proxies)
                    require(proxy->return_notifications_complete(iteration),
                            "iteration returned before notification queues retired");
            }
            stop();
            for (auto& proxy : proxies) proxy->check_forwarding_error();
        }
    };

    static void test_policy_mismatch() {
        for (int mismatch = 0; mismatch < 3; ++mismatch) {
            auto owner_cfg = config("policy");
            owner_cfg.completion_timeout_ms = 30;
            if (mismatch == 1) owner_cfg.nvlink_return_local_batch_sync_enabled = false;
            Proxy owner(owner_cfg);
            owner.initialize_nvlink_forward_notifications();
            auto reader_cfg = owner_cfg;
            reader_cfg.nvlink_return_local_batch_sync_enabled = mismatch != 0;
            if (mismatch == 2) {
                const uint32_t old_version = 5;
                std::memcpy(reinterpret_cast<char*>(owner.nvlink_forward_notification_header_) +
                                sizeof(uint64_t), &old_version, sizeof(old_version));
            }
            Proxy reader(reader_cfg);
            rejects([&] { reader.prepare_forwarding_notification_destinations(); },
                    "mismatch");
            require(reader.forwarding_notification_destinations_.empty(),
                    "mismatched notification policy retained a writable queue");
        }
    }

    struct BarrierFixture {
        ProxyConfig cfg = config("barrier");
        std::array<std::unique_ptr<Proxy>, world> proxies;

        explicit BarrierFixture(int node_rank = 0, int num_nodes = 1) {
            cfg.node_rank = node_rank;
            cfg.num_nodes = num_nodes;
            for (int gpu = 0; gpu < world; ++gpu) {
                cfg.local_gpu_index = gpu;
                proxies[gpu] = std::make_unique<Proxy>(cfg);
                for (int peer = 0; peer < world; ++peer) {
                    if (peer == gpu) continue;
                    Proxy::ForwardDestinationState destination;
                    destination.gpu_index = peer;
                    proxies[gpu]->forwarding_destinations_.push_back(destination);
                }
                proxies[gpu]->initialize_nvlink_forward_notifications();
            }
            for (auto& proxy : proxies) proxy->prepare_forwarding_notification_destinations();
        }

        void configure(int changed_gpu = -1, int changed_field = -1) {
            for (int gpu = 0; gpu < world; ++gpu) {
                const int field = gpu == changed_gpu ? changed_field : -1;
                proxies[gpu]->configure_local_nvlink_return_batch_sync(
                    groups + (field == 0), width + (field == 1), tile + (field == 2),
                    group_tiles + (field == 3), 1 + (field == 4));
            }
        }
    };

    static void test_barrier_rounds_and_iterations() {
        BarrierFixture f;
        f.configure();
        auto poll = [&](int gpu, uint64_t iteration, int group) {
            return f.proxies[gpu]->poll_local_nvlink_return_batch_start(iteration, group);
        };
        rejects([&] { poll(0, 0, 1); }, "skips a group");
        require(!poll(0, 0, 0) && !poll(2, 0, 0), "partial arrivals released group zero");
        rejects([&] { poll(0, 0, 2); }, "skips a group");
        rejects([&] { poll(0, 1, 0); }, "skips a group");
        require(poll(1, 0, 0) && poll(0, 0, 0) && poll(2, 0, 0),
                "all group-zero arrivals failed to release the barrier");
        for (int group = 1; group < groups; ++group) {
            require(!poll(2, 0, group) && !poll(1, 0, group),
                    "previous group arrivals satisfied the next group");
            require(poll(0, 0, group) && poll(1, 0, group) && poll(2, 0, group),
                    "complete return group failed to release");
        }
        rejects([&] { poll(1, 0, 1); }, "stale");
        require(!poll(1, 1, 0), "previous iteration satisfied the next iteration's first group");
        // A source that already entered a later iteration has necessarily
        // passed the old final group; a delayed old poll remains satisfiable.
        require(poll(0, 0, groups - 1), "next-iteration arrival stranded an old final group");
        require(!poll(0, 1, 0) && poll(2, 1, 0) && poll(0, 1, 0) && poll(1, 1, 0),
                "second iteration did not require fresh arrivals from every source");
        rejects([&] { poll(0, 0, 0); }, "stale");
        rejects([&] { poll(0, 1, groups); }, "state/iteration/group");
        rejects([&] { poll(0, UINT64_MAX, 0); }, "state/iteration/group");
    }

    static void test_configuration_and_geometry() {
        for (int field = 0; field < 5; ++field) {
            BarrierFixture f;
            f.configure(1, field);
            rejects([&] { f.proxies[0]->poll_local_nvlink_return_batch_start(0, 0); },
                    "geometry mismatch");
        }
        BarrierFixture delayed;
        for (int gpu : {0, 2})
            delayed.proxies[gpu]->configure_local_nvlink_return_batch_sync(groups, width, tile, group_tiles, 1);
        require(!delayed.proxies[0]->poll_local_nvlink_return_batch_start(0, 0),
                "an unconfigured producer was omitted from the barrier");
        delayed.proxies[1]->configure_local_nvlink_return_batch_sync(groups, width, tile, group_tiles, 1);
        require(!delayed.proxies[1]->poll_local_nvlink_return_batch_start(0, 0) &&
                    delayed.proxies[2]->poll_local_nvlink_return_batch_start(0, 0) &&
                    delayed.proxies[0]->poll_local_nvlink_return_batch_start(0, 0),
                "late peer configuration failed to join the return group");
        rejects([&] {
            delayed.proxies[0]->configure_local_nvlink_return_batch_sync(groups, width, tile, group_tiles, 1);
        }, "configuration");
        // A node in a two-node job needs exactly its own local producers;
        // matching GPU ranks on the remote node are not barrier participants.
        BarrierFixture two_nodes(1, 2);
        two_nodes.configure();
        require(!two_nodes.proxies[1]->poll_local_nvlink_return_batch_start(0, 0) &&
                    !two_nodes.proxies[0]->poll_local_nvlink_return_batch_start(0, 0) &&
                    two_nodes.proxies[2]->poll_local_nvlink_return_batch_start(0, 0) &&
                    two_nodes.proxies[1]->poll_local_nvlink_return_batch_start(0, 0),
                "two-node return barrier depended on a remote proxy");
    }

    static void test_timeout_cancel_and_shutdown() {
        {
            BarrierFixture f;
            f.configure();
            require(!f.proxies[0]->poll_local_nvlink_return_batch_start(0, 0),
                    "missing peers unexpectedly released the timeout test");
            // Move only the host timeout origin, avoiding timing-sensitive
            // sleeps while exercising the real timeout/error branch.
            f.proxies[0]->return_batch_wait_start_ -=
                std::chrono::milliseconds(f.cfg.completion_timeout_ms + 1);
            rejects([&] { f.proxies[0]->poll_local_nvlink_return_batch_start(0, 0); },
                    "timed out waiting for return batch-start barrier");
        }
        for (int mode = 0; mode < 4; ++mode) {
            BarrierFixture f;
            f.configure();
            require(!f.proxies[0]->poll_local_nvlink_return_batch_start(0, 0),
                    "missing peers unexpectedly released the cancellation test");
            if (mode == 0) f.proxies[0]->cancel_local_nvlink_return_batch_sync();
            if (mode == 1) f.proxies[1]->cancel_local_nvlink_return_batch_sync();
            if (mode == 2) f.proxies[1]->set_forwarding_error("injected return producer failure");
            if (mode == 3) f.proxies[1]->shutdown();
            rejects([&] { f.proxies[0]->poll_local_nvlink_return_batch_start(0, 0); }, "cancelled");
        }
    }
};
}  // namespace rdma_proxy

int main() {
    try {
        rdma_proxy::Logger::instance().set_level(rdma_proxy::LogLevel::kError);
        rdma_proxy::ReturnNotificationsTestAccess::test_policy_mismatch();
        rdma_proxy::ReturnNotificationsTestAccess::test_barrier_rounds_and_iterations();
        rdma_proxy::ReturnNotificationsTestAccess::test_configuration_and_geometry();
        rdma_proxy::ReturnNotificationsTestAccess::test_timeout_cancel_and_shutdown();
        rdma_proxy::ReturnNotificationsTestAccess::Fixture(true).exercise();
        rdma_proxy::ReturnNotificationsTestAccess::Fixture(false).exercise();
        std::cout << "return batch synchronization checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
