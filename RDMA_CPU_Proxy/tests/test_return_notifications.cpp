#include "proxy.hpp"
#include "logging.hpp"

#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> void rejects(F action) {
    bool failed = false;
    try { action(); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "expected rejection");
}

void test_progress() {
    rdma_proxy::ReturnGroupProgress progress;
    progress.configure(3, 1, 3);
    rejects([&] { progress.prefix(0); });
    require(!progress.consume(0, 0, 0), "unprepared notification was consumed");
    progress.reset(0);
    progress.consume(0, 0, 0);
    progress.consume(0, 0, 1);
    require(progress.prefix(0) == 0, "fast peer advanced aggregate prefix");
    progress.consume(2, 0, 0);
    require(progress.prefix(0) == 1, "aggregate prefix did not advance");
    rejects([&] { progress.consume(2, 0, 0); }); // Duplicate.
    rejects([&] { progress.consume(2, 0, 2); }); // Gap.
    rejects([&] { progress.consume(1, 0, 0); }); // Self.
    rejects([&] { progress.reset(1); });
    progress.consume(2, 0, 1);
    progress.consume(2, 0, 2);
    require(progress.prefix(0) == 2, "missing last peer was ignored");
    progress.consume(0, 0, 2);
    require(progress.prefix(0) == 3, "last group did not complete");
    require(!progress.consume(0, 1, 0), "future iteration was consumed");
    progress.reset(1);
    require(progress.prefix(1) == 0, "iteration prefix not reset");
    rejects([&] { progress.consume(0, 0, 0); });
    rejects([&] { progress.prefix(0); });
}
} // namespace

namespace rdma_proxy {
// Exercise the actual receiver -> dispatcher -> shared-memory peer queue ->
// forwarding submitter path, with CUDA copies mocked on CPU memory. No NIC/GPU.
struct ReturnNotificationsTestAccess {
    static void exercise(bool atomic_ring, bool enabled, bool publish_gpu = false) {
        constexpr int world = 3, groups = 3, width = 6;
        using Buffer = std::array<unsigned char, width>;
        std::array<std::array<Buffer, world>, world> src{}, dst{};
        std::array<int64_t, world> rows{1, 1, 0}; // Last producer has empty outputs.
        std::array<std::unique_ptr<Proxy>, world> proxies;
        std::array<int32_t*, world> flags{};
        ProxyConfig config;
        config.mock_mode = true;
        config.num_gpus_per_node = world;
        config.num_nodes = 1;
        config.num_iterations = 2;
        config.router_routing_enabled = true;
        config.nvlink_forward_completion_notifications_enabled = true;
        config.nvlink_return_completion_notifications_enabled = enabled;
        config.nvlink_return_ready_gpu_publication_enabled = publish_gpu;
        config.nvlink_forward_preparation_enabled = true;
        config.nvlink_forward_submit_epilogue_enabled = true;
        config.nvlink_forward_prepared_atomic_ring_enabled = atomic_ring;
        // Exactly one iteration fits: detects any missed queue retirement.
        config.nvlink_forward_notification_queue_depth = groups;
        config.completion_timeout_ms = 5000;
        config.local_iteration_sync_run_id = "test_return_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        auto stop = [&] {
            std::array<std::thread, world> threads;
            for (int gpu = 0; gpu < world; ++gpu) {
                if (proxies[gpu]) threads[gpu] = std::thread([&, gpu] {
                    try { proxies[gpu]->shutdown(); } catch (...) {}
                });
            }
            for (auto& thread : threads) if (thread.joinable()) thread.join();
        };
        try {
            for (int gpu = 0; gpu < world; ++gpu) {
                config.local_gpu_index = gpu;
                proxies[gpu] = std::make_unique<Proxy>(config);
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
                if (enabled) rejects([&] { proxy.configure_down_ready_trace(groups + 1); });
                flags[gpu] = reinterpret_cast<int32_t*>(proxy.configure_down_ready_trace(groups));
                if (!publish_gpu) rejects([&] { proxy.return_ready_publication_addresses(); });
                std::vector<RdmaProxyReturnCopy> copies;
                for (int peer = 0; peer < world; ++peer) {
                    if (peer == gpu) continue;
                    copies.push_back({reinterpret_cast<uintptr_t>(src[gpu][peer].data()),
                        reinterpret_cast<uintptr_t>(dst[peer][gpu].data()), width, 0, peer});
                }
                if (enabled) {
                    rejects([&] { proxy.configure_return_copies(copies.data(), copies.size(),
                        &rows[gpu], 1, width, 2, 1, 1, false, 0); });
                    rejects([&] { proxy.configure_return_copies(copies.data(), 1,
                        &rows[gpu], 1, width, 2, 1, 1, true, 0); });
                }
                proxy.configure_return_copies(copies.data(), copies.size(), &rows[gpu],
                    1, width, 2, 1, 1, true, 0);
            }
            auto wait = [&](auto ready) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (!ready()) {
                    for (auto& proxy : proxies) proxy->check_forwarding_error();
                    require(std::chrono::steady_clock::now() < deadline, "return notification wait timed out");
                    std::this_thread::yield();
                }
            };
            for (uint64_t iteration = 0; iteration < 2; ++iteration) {
                for (int gpu = 0; gpu < world; ++gpu) {
                    for (int peer = 0; peer < world; ++peer) {
                        src[gpu][peer].fill(static_cast<unsigned char>(10 + gpu + iteration * 10));
                        dst[gpu][peer].fill(255);
                    }
                }
                for (int gpu = 0; gpu < world; ++gpu) {
                    auto& proxy = *proxies[gpu];
                    if (enabled) proxy.return_group_progress_.reset(iteration);
                    proxy.down_ready_trace_.reset(iteration);
                    proxy.arm_return_copies(iteration);
                    if (publish_gpu && iteration == 0) {
                        const auto addresses = proxy.return_ready_publication_addresses();
                        require(addresses.first != addresses.second, "host and device flags alias");
                        require(__atomic_load_n(reinterpret_cast<int32_t*>(addresses.first), __ATOMIC_ACQUIRE) == 0 &&
                                __atomic_load_n(reinterpret_cast<int32_t*>(addresses.second), __ATOMIC_ACQUIRE) == 0,
                                "return-ready flags were not initialized to zero");
                    }
                    if (enabled && iteration == 1 && gpu == 0) {
                        // A whole future iteration can arrive before peers
                        // reset. Their forwarding threads must leave it queued.
                        __atomic_store_n(flags[0], groups, __ATOMIC_RELEASE);
                        wait([&] { return proxy.return_notifications_sent_.load() == 2 * groups; });
                        require(proxies[1]->return_ready_prefix(0) == groups,
                                "early next-iteration message corrupted old prefix");
                    }
                }
                __atomic_store_n(flags[0], groups, __ATOMIC_RELEASE);
                __atomic_store_n(flags[1], groups, __ATOMIC_RELEASE);
                if (enabled) {
                    // GPU 2 receives both peers' groups but has not sent its own.
                    wait([&] { return proxies[2]->return_ready_prefix(iteration) == groups; });
                    require(proxies[0]->return_ready_prefix(iteration) == 0,
                            "aggregate ignored empty producer's notifications");
                    __atomic_store_n(flags[2], 1, __ATOMIC_RELEASE);
                    wait([&] { return proxies[0]->return_ready_prefix(iteration) == 1; });
                    if (publish_gpu) {
                        const auto addresses = proxies[0]->return_ready_publication_addresses();
                        require(__atomic_load_n(reinterpret_cast<int32_t*>(addresses.second), __ATOMIC_ACQUIRE) == 1,
                                "partial aggregate prefix was not published to mock GPU");
                    }
                    require(proxies[1]->return_ready_prefix(iteration) <= 1, "prefix advanced too far");
                }
                __atomic_store_n(flags[2], groups, __ATOMIC_RELEASE);
                for (auto& proxy : proxies) proxy->read_down_ready_trace(iteration);
                for (int gpu = 0; gpu < world; ++gpu) {
                    for (int peer = 0; peer < world; ++peer) {
                        if (peer == gpu) continue;
                        for (auto byte : dst[gpu][peer])
                            require(byte == (rows[peer] ? 10 + peer + iteration * 10 : 255),
                                    "returned bytes differ after ready prefix");
                    }
                    if (enabled) require(proxies[gpu]->return_notifications_complete(iteration),
                                         "notifications not drained at iteration completion");
                    else {
                        rejects([&] { proxies[gpu]->return_ready_prefix(iteration); });
                        require(proxies[gpu]->return_notifications_sent_.load() == 0,
                                "disabled feature sent notifications");
                    }
                    if (publish_gpu) {
                        const auto addresses = proxies[gpu]->return_ready_publication_addresses();
                        require(__atomic_load_n(reinterpret_cast<int32_t*>(addresses.first), __ATOMIC_ACQUIRE) == groups &&
                                __atomic_load_n(reinterpret_cast<int32_t*>(addresses.second), __ATOMIC_ACQUIRE) == groups,
                                "final returned prefix missing from host or mock GPU flag");
                    }
                }
            }
            stop();
            for (auto& proxy : proxies) proxy->check_forwarding_error();
        } catch (...) {
            stop();
            throw;
        }
    }
};
} // namespace rdma_proxy

int main() {
    try {
        test_progress();
        rdma_proxy::ReturnNotificationsTestAccess::exercise(true, true);
        rdma_proxy::ReturnNotificationsTestAccess::exercise(false, true);
        rdma_proxy::ReturnNotificationsTestAccess::exercise(true, false);
        rdma_proxy::ReturnNotificationsTestAccess::exercise(true, true, true);
        rdma_proxy::ReturnNotificationsTestAccess::exercise(false, true, true);
        std::cout << "return notification checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
