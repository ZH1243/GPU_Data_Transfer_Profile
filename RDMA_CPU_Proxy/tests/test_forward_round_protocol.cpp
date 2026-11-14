#include "logging.hpp"
#include "proxy.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rdma_proxy {
// This executable uses the existing friend seam; production notification
// objects stay opaque. No transport or CUDA operation is mocked by the test:
// these are the actual POSIX queue setup and receiver epoch methods.
struct PreparedForwardingTestAccess {
    static void require(bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    }

    template <typename Function>
    static void reject(Function&& function, const std::string& text) {
        try { function(); }
        catch (const std::runtime_error& error) {
            if (std::string(error.what()).find(text) != std::string::npos) return;
            throw std::runtime_error("unexpected protocol error: " + std::string(error.what()));
        }
        throw std::runtime_error("expected protocol rejection: " + text);
    }

    static ProxyConfig config() {
        ProxyConfig config;
        config.mock_mode = true;
        config.node_rank = 0;
        config.num_nodes = 1;
        config.num_gpus_per_node = 2;
        config.local_gpu_index = 0;
        config.nvlink_forward_completion_notifications_enabled = true;
        config.nvlink_forward_notification_scheduling_mode = "complete_round";
        config.nvlink_forward_notification_queue_depth = 1;
        config.completion_timeout_ms = 40;
        config.local_iteration_sync_run_id = "test_round_protocol_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        return config;
    }

    static void test_policy_and_context_mismatch() {
        for (int mismatch = 0; mismatch < 6; ++mismatch) {
            auto owner_config = config();
            if (mismatch == 1)
                owner_config.nvlink_forward_notification_scheduling_mode = "per_notification";
            Proxy owner(owner_config);
            owner.initialize_nvlink_forward_notifications();
            auto reader_config = owner_config;
            if (mismatch == 0)
                reader_config.nvlink_forward_notification_scheduling_mode = "per_notification";
            if (mismatch == 1)
                reader_config.nvlink_forward_notification_scheduling_mode = "complete_round";
            if (mismatch == 2) reader_config.router_local_input_staging_enabled = true;
            if (mismatch == 5) reader_config.nvlink_forward_notification_scheduling_diagnostics = true;
            // The initial magic/version prefix is stable across v4 through v6.
            // Alter only that wire field to simulate an older live receiver.
            if (mismatch == 3) {
                const uint32_t old_version = 4;
                std::memcpy(reinterpret_cast<char*>(owner.nvlink_forward_notification_header_) +
                                sizeof(uint64_t), &old_version, sizeof(old_version));
                // A newer reader can expect a larger segment. It must still
                // reject the old live protocol from the fixed prefix.
                reader_config.nvlink_forward_notification_queue_depth = 8;
            }
            if (mismatch == 4) {
                // The fixed header has magic, version, initialized, then node.
                const int32_t wrong_node = 1;
                std::memcpy(reinterpret_cast<char*>(owner.nvlink_forward_notification_header_) +
                                sizeof(uint64_t) + 2 * sizeof(uint32_t),
                            &wrong_node, sizeof(wrong_node));
            }
            Proxy reader(reader_config);
            reject([&] { reader.prepare_forwarding_notification_destinations(); },
                   "mismatch");
            require(reader.forwarding_notification_destinations_.empty(),
                    "mismatched shared-memory metadata was retained for publication");
        }
        // A matching pair must actually map; otherwise rejection-only tests
        // could pass because initialization never produced a usable queue.
        auto matching = config();
        Proxy owner(matching), reader(matching);
        owner.initialize_nvlink_forward_notifications();
        reader.prepare_forwarding_notification_destinations();
        require(reader.forwarding_notification_destinations_.size() == 1,
                "matching notification protocol could not be mapped");
    }

    static void test_live_old_owner_is_not_unlinked() {
#if defined(__unix__) || defined(__APPLE__)
        auto owner_config = config();
        Proxy owner(owner_config);
        owner.initialize_nvlink_forward_notifications();
        const uint32_t old_version = 4;
        std::memcpy(reinterpret_cast<char*>(owner.nvlink_forward_notification_header_) +
                        sizeof(uint64_t), &old_version, sizeof(old_version));
        struct stat before {};
        require(fstat(owner.nvlink_forward_notification_fd_, &before) == 0,
                "could not inspect the live notification queue");
        {
            auto contender_config = owner_config;
            // The old segment is smaller than the new requested layout. Its
            // live PID must still be recognized from the fixed header prefix.
            contender_config.nvlink_forward_notification_queue_depth = 8;
            Proxy contender(contender_config);
            reject([&] { contender.initialize_nvlink_forward_notifications(); }, "live owner");
        } // Rejected contender cleanup must not unlink another process's name.
        const int reopened = shm_open(owner.nvlink_forward_notification_name_.c_str(), O_RDWR, 0600);
        require(reopened >= 0, "rejected contender unlinked the live owner's queue");
        struct stat after {};
        const bool inspected = fstat(reopened, &after) == 0;
        close(reopened);
        require(inspected && before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
                    before.st_size == after.st_size,
                "rejected contender replaced the live owner's queue");
#endif
    }

    static void test_receiver_epochs() {
        auto empty_config = config();
        empty_config.router_local_input_staging_enabled = false;
        Proxy empty(empty_config);
        empty.initialize_forwarding_rounds();
        empty.begin_forwarding_round_iteration(); // Nothing requested yet.
        require(empty.forwarding_round_completed_.load() == 0, "unrequested epoch was completed");
        for (uint64_t epoch = 1; epoch <= 3; ++epoch) {
            empty.forwarding_round_requested_.store(epoch, std::memory_order_release);
            empty.begin_forwarding_round_iteration();
            require(empty.forwarding_round_completed_.load(std::memory_order_acquire) == epoch,
                    "no-context iteration did not acknowledge completion");
            empty.begin_forwarding_round_iteration(); // Idempotent receiver poll.
        }

        auto pending_config = config();
        pending_config.router_local_input_staging_enabled = true;
        Proxy pending(pending_config);
        pending.initialize_forwarding_rounds();
        pending.forwarding_round_requested_.store(1, std::memory_order_release);
        pending.begin_forwarding_round_iteration();
        require(pending.forwarding_round_completed_.load() == 0,
                "empty queues incorrectly completed a context with missing source markers");
        pending.forwarding_round_requested_.store(2, std::memory_order_release);
        reject([&] { pending.begin_forwarding_round_iteration(); }, "discard pending work");
        require(pending.forwarding_round_completed_.load() == 0,
                "failed pending reset acknowledged unfinished work");
    }

    static void test_next_epoch_queue_does_not_hold_previous_drain() {
        auto common = config();
        common.router_local_input_staging_enabled = true;
        common.completion_timeout_ms = 2000;
        std::array<std::unique_ptr<Proxy>, 2> proxies;
        for (int gpu = 0; gpu < 2; ++gpu) {
            auto cfg = common;
            cfg.local_gpu_index = gpu;
            proxies[gpu] = std::make_unique<Proxy>(cfg);
            proxies[gpu]->initialize_nvlink_forward_notifications();
        }
        std::exception_ptr work_error;
        std::array<std::exception_ptr, 2> stop_errors;
        try {
            for (int gpu = 0; gpu < 2; ++gpu) {
                auto& proxy = *proxies[gpu];
                // This component fixture only sends zero-source phase ends;
                // no token-buffer IPC allocation or GPU operation is needed.
                Proxy::ForwardDestinationState destination;
                destination.gpu_index = 1 - gpu;
                proxy.forwarding_destinations_.push_back(std::move(destination));
                proxy.prepare_forwarding_notification_destinations();
                proxy.initialize_nvlink_forward_notification_dispatch();
                proxy.initialize_forwarding_rounds();
                proxy.forwarding_round_requested_.store(1, std::memory_order_release);
                proxy.nvlink_forward_notification_dispatch_thread_ =
                    std::thread(&Proxy::nvlink_forward_notification_dispatch_loop, &proxy);
                proxy.nvlink_forward_notification_thread_ =
                    std::thread(&Proxy::nvlink_forward_notification_loop, &proxy);
            }
            auto drain_both = [&](uint64_t iteration) {
                std::array<std::exception_ptr, 2> errors;
                std::array<std::thread, 2> drains;
                for (int gpu = 0; gpu < 2; ++gpu) {
                    drains[gpu] = std::thread([&, gpu] {
                        try { proxies[gpu]->drain_forwarding_iteration_notifications(iteration); }
                        catch (...) { errors[gpu] = std::current_exception(); }
                    });
                }
                for (auto& drain : drains) drain.join();
                for (auto error : errors) if (error) std::rethrow_exception(error);
            };
            for (auto& proxy : proxies) proxy->enqueue_empty_forwarding_phases(0);
            drain_both(0);
            for (auto& proxy : proxies)
                require(proxy->forwarding_round_completed_.load(std::memory_order_acquire) == 1,
                        "first epoch drain did not collect all empty-source markers");

            // A fast source queues E1 after publishing its E0 watermark. Both
            // receivers still have E0 prepared and must leave E1 in their
            // depth-one queues. Their old epoch is nevertheless fully drained.
            proxies[0]->enqueue_empty_forwarding_phases(1);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (proxies[0]->nvlink_forward_notifications_sent_.load() != 4) {
                proxies[0]->check_forwarding_error();
                require(std::chrono::steady_clock::now() < deadline,
                        "next-epoch marker could not enter the bounded queue");
                std::this_thread::yield();
            }
            for (auto& proxy : proxies)
                require(proxy->nvlink_forward_notifications_received_.load() == 2 &&
                            proxy->forwarding_round_completed_.load(std::memory_order_acquire) == 1,
                        "future-epoch marker bypassed receiver preparation");
            // This call deterministically sees a queued next-epoch marker.
            // Requiring tail==head for E0 would time out here.
            proxies[1]->drain_forwarding_iteration_notifications(0);

            for (auto& proxy : proxies)
                proxy->forwarding_round_requested_.store(2, std::memory_order_release);
            const auto consume_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (proxies[1]->nvlink_forward_notifications_received_.load() != 3) {
                proxies[1]->check_forwarding_error();
                require(std::chrono::steady_clock::now() < consume_deadline,
                        "prepared receiver did not consume the queued next-epoch marker");
                std::this_thread::yield();
            }
            require(proxies[1]->forwarding_round_completed_.load(std::memory_order_acquire) == 1,
                    "one source marker or old watermark completed the next epoch prematurely");
            proxies[1]->enqueue_empty_forwarding_phases(1);
            drain_both(1);
            for (auto& proxy : proxies) {
                require(proxy->forwarding_round_completed_.load(std::memory_order_acquire) == 2,
                        "next epoch failed after all source markers arrived");
                proxy->check_forwarding_error();
            }
        } catch (...) {
            work_error = std::current_exception();
            for (auto& proxy : proxies) proxy->set_forwarding_error("protocol fixture cancellation");
        }
        // Each receiver needs both senders' shutdown markers, so shut the
        // independent proxy components down concurrently, as real workers do.
        std::array<std::thread, 2> stopping;
        for (int gpu = 0; gpu < 2; ++gpu) {
            stopping[gpu] = std::thread([&, gpu] {
                try { proxies[gpu]->stop_forwarding_thread(); }
                catch (...) { stop_errors[gpu] = std::current_exception(); }
            });
        }
        for (auto& stop : stopping) stop.join();
        if (work_error) std::rethrow_exception(work_error);
        for (auto error : stop_errors) if (error) std::rethrow_exception(error);
        for (auto& proxy : proxies) proxy->check_forwarding_error();
    }
};
} // namespace rdma_proxy

int main() {
    try {
        rdma_proxy::Logger::instance().set_level(rdma_proxy::LogLevel::kError);
        rdma_proxy::PreparedForwardingTestAccess::test_policy_and_context_mismatch();
        rdma_proxy::PreparedForwardingTestAccess::test_live_old_owner_is_not_unlinked();
        rdma_proxy::PreparedForwardingTestAccess::test_receiver_epochs();
        rdma_proxy::PreparedForwardingTestAccess::test_next_epoch_queue_does_not_hold_previous_drain();
        std::cout << "forwarding-round protocol checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
