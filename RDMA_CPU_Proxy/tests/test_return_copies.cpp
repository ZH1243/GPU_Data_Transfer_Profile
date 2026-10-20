#include "down_ready_trace.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

using rdma_proxy::DownReadyTrace;
void require(bool condition) {
    if (!condition) throw std::runtime_error("return-copy assertion failed");
}
template <typename F> void rejects(F action) {
    bool failed = false;
    try { action(); } catch (const std::runtime_error&) { failed = true; }
    require(failed);
}

// Four sources: two segments per peer, uneven counts, an empty source and
// capacity larger than the physical valid prefix. Values identify every byte.
void exercise(bool tiled, bool streaming) {
    constexpr int width = 10, tile = 4, group_tiles = 2, groups = 2, capacity = 80;
    std::array<int64_t, 4> rows{3, 1, 0, 2};
    std::array<std::array<unsigned char, capacity>, 4> src{}, dst{};
    std::array<RdmaProxyReturnCopy, 4> copies{};
    for (int i = 0; i < 4; ++i) {
        copies[i] = {reinterpret_cast<uintptr_t>(src[i].data()),
                     reinterpret_cast<uintptr_t>(dst[i].data()), capacity,
                     static_cast<uint64_t>(i), i / 2};
    }
    DownReadyTrace trace;
    auto* flag = reinterpret_cast<int32_t*>(trace.configure(0, groups, 1000, true));
    trace.configure_returns(copies.data(), copies.size(), rows.data(), rows.size(),
                            width, tiled ? tile : 0, group_tiles, 1, streaming, 0);
    std::vector<std::pair<int32_t, double>> observations;
    for (int iteration = 0; iteration < 3; ++iteration) {
        if (iteration == 1) rows = {1, 0, 4, 3};
        if (iteration == 2) rows = {0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) {
            dst[i].fill(255);
            for (int j = 0; j < capacity; ++j) src[i][j] = (iteration * 71 + i * 17 + j) % 250;
        }
        trace.reset(iteration);
        rejects([&] { trace.arm_returns(iteration + 1); });
        // Even an already-published prefix cannot access last generation's event.
        __atomic_store_n(flag, 1, __ATOMIC_RELEASE);
        trace.poll();
        require(dst[0][0] == 255);
        require(!trace.read(iteration, observations));
        trace.arm_returns(iteration);
        rejects([&] { trace.arm_returns(iteration); });
        trace.poll();
        for (int i = 0; i < 4; ++i) {
            const auto partial = streaming ? rows[i] * group_tiles * tile : 0;
            for (int j = 0; j < capacity; ++j)
                require(dst[i][j] == (j < partial ? src[i][j] : 255));
        }
        require(!trace.read(iteration, observations));
        __atomic_store_n(flag, groups, __ATOMIC_RELEASE);
        trace.poll();
        require(trace.read(iteration, observations));
        for (int i = 0; i < 4; ++i) {
            const auto valid_bytes = rows[i] * (tiled ? 12 : width);
            for (int j = 0; j < capacity; ++j)
                require(dst[i][j] == (j < valid_bytes ? src[i][j] : 255));
        }
        // Repeated polls must not resubmit completed groups.
        dst[0][0] = 254;
        trace.poll();
        require(dst[0][0] == 254);
    }
    trace.drain_returns();
}

void jumps_and_bounds() {
    std::array<unsigned char, 32> src{}, dst{};
    for (int i = 0; i < 32; ++i) src[i] = i + 1;
    int64_t rows = 2;
    RdmaProxyReturnCopy copy{reinterpret_cast<uintptr_t>(src.data()),
                             reinterpret_cast<uintptr_t>(dst.data()), 32, 0, 1};
    DownReadyTrace trace;
    auto* flag = reinterpret_cast<int32_t*>(trace.configure(0, 4, 1000, true));
    trace.configure_returns(&copy, 1, &rows, 1, 16, 4, 1, 1, true, 0);
    trace.reset(0);
    trace.arm_returns(0);
    __atomic_store_n(flag, 4, __ATOMIC_RELEASE);
    trace.poll();
    require(src == dst);  // All skipped intermediate groups were copied.
    trace.reset(1);
    rows = 3;  // Packed bytes exceed both allocation capacities.
    trace.arm_returns(1);
    __atomic_store_n(flag, 1, __ATOMIC_RELEASE);
    rejects([&] { trace.poll(); });
    trace.cancel();
    std::vector<std::pair<int32_t, double>> observations;
    rejects([&] { trace.read(1, observations); });
}

int main() {
    try {
        exercise(false, false);
        exercise(true, false);
        exercise(true, true);
        jumps_and_bounds();
        std::cout << "return-copy checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
