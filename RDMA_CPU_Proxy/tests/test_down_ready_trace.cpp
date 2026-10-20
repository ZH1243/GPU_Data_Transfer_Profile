#include "down_ready_trace.hpp"
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

int main() {
    try {
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
