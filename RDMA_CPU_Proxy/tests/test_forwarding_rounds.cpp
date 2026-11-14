#include "forwarding_rounds.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using rdma_proxy::ForwardingRound;
using rdma_proxy::ForwardingRoundCollector;
using rdma_proxy::ForwardingRoundPhase;
using Kind = ForwardingRoundPhase::Kind;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void reject(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, message);
}

void test_arrival_permutations() {
    std::array<std::size_t, 3> order{0, 1, 2};
    do {
        ForwardingRoundCollector<int> collector(3, {{Kind::kRemote, 7}}, 3);
        collector.begin_iteration(12);
        ForwardingRound<int> ready;
        for (std::size_t i = 0; i < order.size(); ++i) {
            const auto gpu = order[i];
            collector.add_data(12, 0, gpu, 0, static_cast<int>(gpu + 10));
            require(!collector.pop_ready(ready), "data alone scheduled an incomplete round");
            collector.complete_batch(12, 0, gpu, 0, 100 + i * 10);
            collector.complete_phase(12, 0, gpu, 1);
            if (i + 1 < order.size())
                require(!collector.pop_ready(ready), "round scheduled before every GPU completed");
        }
        require(collector.pop_ready(ready), "fully completed round was not released");
        require(ready.iteration == 12 && ready.phase_index == 0 && ready.ordinal == 0 &&
                    ready.phase.kind == Kind::kRemote && ready.phase.source_node_rank == 7,
                "round identity lost iteration, phase or source node");
        require(ready.payloads == std::vector<int>({10, 11, 12}), "source payload order changed");
        require(ready.first_completion_ns == 100 && ready.last_completion_ns == 120,
                "completion timestamps do not describe the collection");
        require(collector.complete() && collector.pending_batches() == 0,
                "completed round retained work");
        require(!collector.pop_ready(ready), "round released twice");
    } while (std::next_permutation(order.begin(), order.end()));
}

void test_unequal_and_zero_data() {
    ForwardingRoundCollector<int> collector(3, {{Kind::kLocal, 4}}, 6);
    collector.begin_iteration(0);
    ForwardingRound<int> ready;
    // GPU 0 is entirely empty. GPU 1 has a zero-route destination in its first
    // batch and then ends; GPU 2 has three batches and runs far ahead.
    collector.complete_phase(0, 0, 0, 0);
    for (uint64_t round = 0; round < 3; ++round) {
        collector.add_data(0, 0, 2, round, static_cast<int>(20 + round));
        collector.complete_batch(0, 0, 2, round);
        require(!collector.pop_ready(ready), "exhausted source bypassed a live participant");
    }
    collector.complete_phase(0, 0, 2, 3);
    collector.complete_batch(0, 0, 1, 0); // No data still requires completion.
    require(collector.pop_ready(ready) && ready.ordinal == 0 &&
                ready.payloads == std::vector<int>({20}), "zero-route batch did not complete round");
    require(!collector.pop_ready(ready), "absent next batch incorrectly implied exhaustion");
    collector.complete_phase(0, 0, 1, 1);
    for (uint64_t round = 1; round < 3; ++round) {
        require(collector.pop_ready(ready) && ready.ordinal == round &&
                    ready.payloads == std::vector<int>({static_cast<int>(20 + round)}),
                "partial final round or future payload was lost");
    }
    require(collector.complete() && !collector.pop_ready(ready), "unequal sources did not finish");

    // An all-zero-destination round remains a real collection, even though it
    // need not result in any gather rows.
    collector.begin_iteration(1);
    for (std::size_t gpu = 0; gpu < 3; ++gpu) {
        collector.complete_batch(1, 0, gpu, 0);
        collector.complete_phase(1, 0, gpu, 1);
    }
    require(collector.pop_ready(ready) && ready.payloads.empty(), "zero-data round lost its boundary");
    require(collector.complete(), "zero-data iteration failed to finish");
    collector.begin_iteration(2);
    for (std::size_t gpu = 0; gpu < 3; ++gpu) collector.complete_phase(2, 0, gpu, 0);
    require(collector.complete() && !collector.pop_ready(ready), "empty iteration created fake work");
}

void test_interleaved_contexts() {
    ForwardingRoundCollector<int> collector(
        2, {{Kind::kLocal, 1}, {Kind::kRemote, 0}, {Kind::kRemote, 2}}, 12);
    collector.begin_iteration(7);
    ForwardingRound<int> ready;
    // The same source can interleave remote nodes while local collection is
    // incomplete. Their ordinal 0 values must never be mistaken for one round.
    for (std::size_t gpu = 0; gpu < 2; ++gpu) {
        collector.add_data(7, 2, gpu, 0, static_cast<int>(200 + gpu));
        collector.complete_batch(7, 2, gpu, 0);
        collector.complete_phase(7, 2, gpu, 1);
        collector.add_data(7, 1, gpu, 0, static_cast<int>(100 + gpu));
        collector.complete_batch(7, 1, gpu, 0);
        collector.complete_phase(7, 1, gpu, 1);
    }
    require(!collector.pop_ready(ready), "remote phase bypassed incomplete local phase");
    collector.complete_phase(7, 0, 0, 0); // Own early-local notification suppressed.
    collector.add_data(7, 0, 1, 0, 1);
    collector.complete_batch(7, 0, 1, 0);
    require(collector.pop_ready(ready) && ready.phase_index == 0 &&
                ready.payloads == std::vector<int>({1}), "local round used future remote data");
    require(!collector.pop_ready(ready), "phase transition ignored missing exhaustion marker");
    collector.complete_phase(7, 0, 1, 1);
    require(collector.pop_ready(ready) && ready.phase_index == 1 &&
                ready.payloads == std::vector<int>({100, 101}), "remote peer order was not retained");
    require(collector.pop_ready(ready) && ready.phase_index == 2 &&
                ready.payloads == std::vector<int>({200, 201}), "second remote peer lost its identity");
    require(collector.complete(), "interleaved phases did not finish");
}

void test_validation_and_cancellation() {
    reject([] { ForwardingRoundCollector<int> bad(0, {{Kind::kLocal, 0}}, 2); },
           "zero participants accepted");
    reject([] { ForwardingRoundCollector<int> bad(1, {{Kind::kLocal, 0}}, 0); },
           "unbounded/zero pending limit accepted");
    reject([] { ForwardingRoundCollector<int> bad(1, {{Kind::kRemote, -1}}, 2); },
           "invalid source node accepted");
    reject([] { ForwardingRoundCollector<int> bad(
        1, {{Kind::kRemote, 2}, {Kind::kRemote, 2}}, 2); }, "ambiguous phase context accepted");
    ForwardingRoundCollector<int> collector(2, {{Kind::kLocal, 0}}, 2);
    reject([&] { collector.complete_batch(0, 0, 0, 0); }, "record accepted before iteration reset");
    collector.begin_iteration(5);
    reject([&] { collector.begin_iteration(6); }, "reset ignored missing zero-data phase ends");
    reject([&] { collector.complete_batch(4, 0, 0, 0); }, "stale iteration accepted");
    reject([&] { collector.complete_batch(6, 0, 0, 0); }, "future iteration accepted");
    reject([&] { collector.complete_batch(5, 1, 0, 0); }, "unknown phase accepted");
    reject([&] { collector.complete_batch(5, 0, 2, 0); }, "unknown source accepted");
    reject([&] { collector.complete_batch(5, 0, 0, 1); }, "skipped FIFO ordinal accepted");
    collector.add_data(5, 0, 0, 0, 8);
    reject([&] { collector.add_data(5, 0, 0, 0, 9); }, "two data records accepted for one batch");
    reject([&] { collector.complete_phase(5, 0, 0, 0); }, "phase ended with an incomplete transfer");
    collector.complete_batch(5, 0, 0, 0);
    reject([&] { collector.complete_batch(5, 0, 0, 0); }, "duplicate batch accepted");
    collector.complete_batch(5, 0, 0, 1);
    reject([&] { collector.complete_batch(5, 0, 0, 2); }, "pending bound was not enforced");
    require(collector.pending_batches() == 2, "rejected input corrupted pending accounting");
    collector.complete_phase(5, 0, 0, 2);
    reject([&] { collector.complete_phase(5, 0, 0, 2); }, "duplicate phase end accepted");
    reject([&] { collector.add_data(5, 0, 0, 2, 9); }, "late data accepted after phase end");
    reject([&] { collector.begin_iteration(6); }, "reset discarded pending data");
    collector.cancel();
    collector.cancel();
    require(!collector.active() && collector.pending_batches() == 0, "cancel retained pending work");
    reject([&] { collector.complete_phase(5, 0, 1, 0); }, "canceled iteration accepted input");
    reject([&] { collector.begin_iteration(5); }, "canceled identity was reused");
    collector.begin_iteration(6);
    collector.complete_phase(6, 0, 0, 0);
    collector.complete_phase(6, 0, 1, 0);
    require(collector.complete(), "fresh iteration retained canceled state");

    ForwardingRoundCollector<int> no_contexts(2, {}, 2);
    ForwardingRound<int> ready;
    require(!no_contexts.complete(), "unstarted empty collector reported an iteration complete");
    for (uint64_t iteration = 0; iteration < 3; ++iteration) {
        no_contexts.begin_iteration(iteration);
        require(no_contexts.complete() && !no_contexts.pop_ready(ready),
                "no-context iteration waited for nonexistent phase markers");
    }
}

void test_draining_bounded_queues() {
    // Each producer has a depth-one queue. GPU 1 deliberately emits nothing
    // until GPU 0 has enqueued *all* its work. Waiting inside collection rather
    // than draining the transport would deadlock this test.
    struct Record { uint64_t ordinal; bool end; };
    std::array<std::deque<Record>, 2> queues;
    std::mutex mutex;
    std::condition_variable changed;
    bool fast_finished = false, stop = false;
    std::array<std::exception_ptr, 2> errors;
    constexpr uint64_t rounds = 128;
    auto push = [&](std::size_t gpu, Record record) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!changed.wait_for(lock, std::chrono::seconds(5), [&] {
                return stop || queues[gpu].empty();
            })) throw std::runtime_error("bounded source queue was not drained");
        if (stop) return;
        queues[gpu].push_back(record);
        changed.notify_all();
    };
    std::thread fast([&] {
        try {
            for (uint64_t i = 0; i < rounds; ++i) push(0, {i, false});
            push(0, {rounds, true});
            std::lock_guard<std::mutex> lock(mutex);
            fast_finished = true;
            changed.notify_all();
        } catch (...) { errors[0] = std::current_exception(); }
    });
    std::thread slow([&] {
        try {
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (!changed.wait_for(lock, std::chrono::seconds(5), [&] {
                        return stop || fast_finished;
                    })) throw std::runtime_error("fast source could not finish before slow marker");
            }
            push(1, {0, true});
        } catch (...) { errors[1] = std::current_exception(); }
    });
    ForwardingRoundCollector<uint64_t> collector(2, {{Kind::kRemote, 1}}, rounds);
    collector.begin_iteration(0);
    ForwardingRound<uint64_t> ready;
    std::vector<uint64_t> observed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    std::exception_ptr receiver_error;
    try {
        while (!collector.complete()) {
            require(std::chrono::steady_clock::now() < deadline, "bounded queue receiver timed out");
            for (std::size_t gpu = 0; gpu < 2; ++gpu) {
                Record record{};
                bool found = false;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!queues[gpu].empty()) {
                        record = queues[gpu].front();
                        queues[gpu].pop_front();
                        found = true;
                        changed.notify_all();
                    }
                }
                if (!found) continue;
                if (record.end) collector.complete_phase(0, 0, gpu, record.ordinal);
                else {
                    collector.add_data(0, 0, gpu, record.ordinal, record.ordinal);
                    collector.complete_batch(0, 0, gpu, record.ordinal);
                }
                while (collector.pop_ready(ready)) {
                    require(ready.payloads.size() == 1, "backpressure duplicated/lost a route");
                    observed.push_back(ready.payloads.front());
                }
            }
            std::this_thread::yield();
        }
    } catch (...) { receiver_error = std::current_exception(); }
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
        changed.notify_all();
    }
    fast.join();
    slow.join();
    if (receiver_error) std::rethrow_exception(receiver_error);
    for (auto error : errors) if (error) std::rethrow_exception(error);
    require(observed.size() == rounds, "backpressure did not preserve every round exactly once");
    for (uint64_t i = 0; i < rounds; ++i)
        require(observed[i] == i, "per-source FIFO broke under backpressure");
}
} // namespace

int main() {
    try {
        test_arrival_permutations();
        test_unequal_and_zero_data();
        test_interleaved_contexts();
        test_validation_and_cancellation();
        test_draining_bounded_queues();
        std::cout << "forwarding-round collection checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
