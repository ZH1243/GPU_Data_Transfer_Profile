#include "down_ready_trace.hpp"
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using rdma_proxy::DownReadyTrace;

void require(bool value) {
    if (!value) throw std::runtime_error("down-ready trace assertion failed");
}

template <typename F> void rejects(F action) {
    bool failed = false;
    try { action(); } catch (const std::runtime_error&) { failed = true; }
    require(failed);
}

void gated_returns() {
    std::array<unsigned char, 8> src{1, 2, 3, 4, 5, 6, 7, 8}, dst{};
    int64_t rows = 1;
    RdmaProxyReturnCopy copy{reinterpret_cast<uintptr_t>(src.data()),
                             reinterpret_cast<uintptr_t>(dst.data()), dst.size(), 0, 0};
    int allowed = 0;
    std::vector<std::pair<uint64_t, int>> starts, completed;
    DownReadyTrace trace;
    auto* flag = reinterpret_cast<int32_t*>(trace.configure(0, 2, 1000, true));
    trace.configure_returns(
        &copy, 1, &rows, 1, 8, 4, 1, 1, true, 0, 1,
        [&](uint64_t iteration, int group) { completed.emplace_back(iteration, group); },
        [&](uint64_t iteration, int group) {
            starts.emplace_back(iteration, group);
            return group < allowed;
        });
    std::vector<std::pair<int32_t, double>> observations;
    for (uint64_t iteration : {uint64_t{9}, uint64_t{10}}) {
        dst.fill(0);
        allowed = 0;
        starts.clear();
        completed.clear();
        trace.reset(iteration);
        __atomic_store_n(flag, 2, __ATOMIC_RELEASE);
        trace.poll();
        require(starts.empty());  // Waiting for the current down event to be armed.
        trace.arm_returns(iteration);
        trace.poll();
        require(starts == std::vector<std::pair<uint64_t, int>>{{iteration, 0}});
        require(completed.empty() && dst[0] == 0);
        require(!trace.read(iteration, observations));  // Final GPU prefix is insufficient.
        allowed = 1;
        trace.poll();
        require(completed == std::vector<std::pair<uint64_t, int>>{{iteration, 0}});
        require(dst[3] == 4 && dst[4] == 0);
        require(!trace.read(iteration, observations));
        allowed = 2;
        trace.poll();
        require(completed == std::vector<std::pair<uint64_t, int>>{{iteration, 0}, {iteration, 1}});
        require(trace.read(iteration, observations) && src == dst);
        require(observations.size() == 1 && observations[0].first == 2);
    }
    trace.reset(11);
    trace.arm_returns(11);
    dst.fill(0);
    allowed = 0;
    __atomic_store_n(flag, 2, __ATOMIC_RELEASE);
    trace.poll();
    require(starts.back() == std::make_pair(uint64_t{11}, 0));
    const auto start_count = starts.size(), completion_count = completed.size();
    trace.cancel();
    allowed = 2;
    trace.poll();
    require(starts.size() == start_count && completed.size() == completion_count);
    require(dst[0] == 0);
    rejects([&] { trace.read(11, observations); });
    rejects([&] { trace.arm_returns(11); });
    // Teardown drains already-submitted work without waiting for the gate.
    trace.drain_returns();

    DownReadyTrace timeout;
    auto* timeout_flag = reinterpret_cast<int32_t*>(timeout.configure(0, 2, 1, true));
    timeout.configure_returns(&copy, 1, &rows, 1, 8, 4, 1, 1, true, 0, 1, {},
                              [](uint64_t, int) { return false; });
    timeout.reset(0);
    timeout.arm_returns(0);
    __atomic_store_n(timeout_flag, 2, __ATOMIC_RELEASE);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    rejects([&] { timeout.poll(); });
    timeout.cancel();
    require(dst[0] == 0);
}

int main() {
    try {
        gated_returns();
        DownReadyTrace trace;
        auto* flag = reinterpret_cast<int32_t*>(trace.configure(0, 4, 1000, true));
        std::vector<std::pair<int32_t, double>> observations;
        rejects([&] { trace.read(0, observations); });
        rejects([&] { trace.configure(0, 4, 1000, true); });
        for (uint64_t iteration = 0; iteration < 2; ++iteration) {
            trace.reset(iteration);
            require(__atomic_load_n(flag, __ATOMIC_ACQUIRE) == 0);
            rejects([&] { trace.reset(iteration + 1); });
            require(!trace.read(iteration, observations));
            trace.poll();
            __atomic_store_n(flag, 1, __ATOMIC_RELEASE);
            trace.poll();
            trace.poll();  // No duplicate observation.
            __atomic_store_n(flag, 4, __ATOMIC_RELEASE);  // Coalesced prefix jump.
            trace.poll();
            require(trace.read(iteration, observations));
            require(observations.size() == 2);
            require(observations[0].first == 1 && observations[1].first == 4);
            require(observations[1].second >= observations[0].second);
            rejects([&] { trace.read(iteration + 1, observations); });
        }
        trace.reset(2);
        __atomic_store_n(flag, 5, __ATOMIC_RELEASE);
        rejects([&] { trace.poll(); });
        trace.cancel();
        rejects([&] { trace.read(2, observations); });
        trace.reset(3);
        __atomic_store_n(flag, 2, __ATOMIC_RELEASE);
        trace.poll();
        __atomic_store_n(flag, 1, __ATOMIC_RELEASE);
        rejects([&] { trace.poll(); });
        trace.cancel();
        DownReadyTrace timeout;
        timeout.configure(0, 1, 1, true);
        timeout.reset(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        rejects([&] { timeout.poll(); });
        std::cout << "down-ready trace checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
