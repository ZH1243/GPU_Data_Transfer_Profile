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
                            width, tiled ? tile : 0, group_tiles, 1, streaming, 0, 2);
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
    trace.configure_returns(&copy, 1, &rows, 1, 16, 4, 1, 1, true, 0, 0);
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

// A prefix may jump to the final group while peer readiness still advances one
// group at a time. Empty destinations and all-empty producers must participate
// in exactly the same start/completion protocol as nonempty producers.
void gated_groups() {
    constexpr int capacity = 32, groups = 4;
    std::array<std::array<unsigned char, capacity>, 2> src{}, dst{};
    std::array<int64_t, 2> rows{2, 0};
    std::array<RdmaProxyReturnCopy, 2> copies{};
    for (int i = 0; i < 2; ++i) {
        copies[i] = {reinterpret_cast<uintptr_t>(src[i].data()),
                     reinterpret_cast<uintptr_t>(dst[i].data()), capacity,
                     static_cast<uint64_t>(i), i};
        for (int j = 0; j < capacity; ++j) src[i][j] = 1 + i * capacity + j;
    }
    int allowed = 0;
    std::vector<int> starts, completed;
    rdma_proxy::ReturnCopies returns(
        0, true, groups, copies.data(), copies.size(), rows.data(), rows.size(),
        16, 4, 1, 1, true, 0, 2,
        [&](int group) {
            require(group == static_cast<int>(completed.size()));
            // Notification follows this group's copies, including empty batches.
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < capacity; ++j)
                    require(dst[i][j] == (j < rows[i] * 4 * (group + 1) ? src[i][j] : 255));
            completed.push_back(group);
        },
        [&](int group) {
            starts.push_back(group);
            return group < allowed;
        });
    for (int iteration = 0; iteration < 3; ++iteration) {
        if (iteration == 1) rows = {0, 2};
        if (iteration == 2) rows = {0, 0};
        for (auto& output : dst) output.fill(255);
        allowed = 0;
        starts.clear();
        completed.clear();
        returns.reset();
        require(!returns.poll(groups));  // Unarmed: do not announce a batch.
        require(starts.empty());
        returns.arm();
        require(!returns.poll(0));        // No local readiness: no peer gate.
        require(starts.empty());
        require(!returns.poll(groups));
        require(starts == std::vector<int>{0});
        require(completed.empty());
        for (const auto& output : dst)
            require(std::all_of(output.begin(), output.end(), [](auto byte) { return byte == 255; }));
        rejects([&] { returns.reset(); });
        for (allowed = 1; allowed <= groups; ++allowed) {
            require(returns.poll(groups) == (allowed == groups));
            require(completed.size() == static_cast<std::size_t>(allowed));
            for (int i = 0; i < allowed; ++i) require(completed[i] == i);
            // A repeated unsuccessful poll cannot recopy earlier groups or
            // report completion for the pending one.
            require(returns.poll(groups) == (allowed == groups));
            require(completed.size() == static_cast<std::size_t>(allowed));
        }
        const auto start_count = starts.size();
        dst[0][0] = 254;
        require(returns.poll(groups));
        require(starts.size() == start_count && dst[0][0] == 254);
    }
    returns.drain();
    rejects([&] {
        rdma_proxy::ReturnCopies invalid(
            0, true, groups, copies.data(), copies.size(), rows.data(), rows.size(),
            16, 4, 1, 1, false, 0, 2, {}, [](int) { return true; });
    });
}

// Each destination reads the preceding destination's output. Only the expected
// cyclic submission order propagates the seed to every destination. This checks
// actual mock copy execution, rather than a separate ordering helper. Distinct
// buffers model two original source-node segments for each destination GPU.
void cyclic_submission_order(int gpu_count, int local_rank, bool tiled,
                             bool streaming, bool sparse) {
    constexpr int max_gpus = 8, segments = 2, capacity = 40;
    constexpr int width = 10, tile = 4, group_tiles = 2, groups = 2;
    constexpr unsigned char sentinel = 255;
    using SegmentBuffers = std::array<std::array<unsigned char, capacity>, segments>;
    std::array<SegmentBuffers, max_gpus> output{};
    SegmentBuffers seed{};
    std::array<int64_t, max_gpus * segments> rows{};
    const int missing_peer = sparse && gpu_count >= 4 ? (local_rank + 1) % gpu_count : -1;
    const int empty_peer = sparse && gpu_count >= 4 ? (local_rank + 3) % gpu_count : -1;
    std::vector<RdmaProxyReturnCopy> copies;
    int previous_peer = -1;
    for (int offset = 1; offset < gpu_count; ++offset) {
        const int peer = (local_rank + offset) % gpu_count;
        if (peer == missing_peer) continue;
        for (int segment = 0; segment < segments; ++segment) {
            const auto row_index = static_cast<std::size_t>(peer * segments + segment);
            rows[row_index] = peer == empty_peer ? 0 : segments - segment;
            const auto* source = previous_peer < 0 ? seed[segment].data()
                                                  : output[previous_peer][segment].data();
            copies.push_back({reinterpret_cast<uintptr_t>(source),
                              reinterpret_cast<uintptr_t>(output[peer][segment].data()),
                              capacity, row_index, peer});
        }
        // Empty peers have no copy that could propagate the chain.
        if (peer != empty_peer) previous_peer = peer;
    }
    // Neither descriptor insertion order nor physical CUDA ordinal determines
    // the local logical-rank rotation used for the return launch sequence.
    std::reverse(copies.begin(), copies.end());
    std::rotate(copies.begin(), copies.begin() + 1, copies.end());
    DownReadyTrace trace;
    auto* flag = reinterpret_cast<int32_t*>(trace.configure(100 + local_rank, groups, 1000, true));
    trace.configure_returns(copies.data(), copies.size(), rows.data(), rows.size(),
                            width, tiled ? tile : 0, group_tiles, 1, streaming, 0, local_rank);
    auto check_prefix = [&](int ready) {
        for (int peer = 0; peer < gpu_count; ++peer) {
            for (int segment = 0; segment < segments; ++segment) {
                const auto valid_rows = rows[peer * segments + segment];
                const auto columns = ready == groups ? (tiled ? 12 : width)
                                                     : (streaming ? tile * group_tiles * ready : 0);
                for (int byte = 0; byte < capacity; ++byte)
                    require(output[peer][segment][byte] ==
                            (byte < valid_rows * columns ? seed[segment][byte] : sentinel));
            }
        }
    };
    for (uint64_t iteration = 0; iteration < 2; ++iteration) {
        for (auto& peer : output)
            for (auto& segment : peer) segment.fill(sentinel);
        for (int segment = 0; segment < segments; ++segment)
            for (int byte = 0; byte < capacity; ++byte)
                seed[segment][byte] = 1 + (iteration * 83 + segment * 43 + byte) % 250;
        trace.reset(iteration);
        trace.arm_returns(iteration);
        __atomic_store_n(flag, 1, __ATOMIC_RELEASE);
        trace.poll();
        check_prefix(1);
        // The final group contains one tile rather than group_tiles tiles.
        __atomic_store_n(flag, groups, __ATOMIC_RELEASE);
        trace.poll();
        check_prefix(groups);
        std::vector<std::pair<int32_t, double>> observations;
        require(trace.read(iteration, observations));
    }
    trace.drain_returns();
}

void invalid_local_rank() {
    std::array<unsigned char, 8> src{}, dst{};
    int64_t rows = 1;
    RdmaProxyReturnCopy copy{reinterpret_cast<uintptr_t>(src.data()),
                             reinterpret_cast<uintptr_t>(dst.data()), dst.size(), 0, 1};
    for (const int invalid_rank : {-1, 1}) {
        rejects([&] {
            rdma_proxy::ReturnCopies invalid(
                0, true, 2, &copy, 1, &rows, 1, 8, 4, 1, 1, true, 0, invalid_rank);
        });
    }
}

int main() {
    try {
        exercise(false, false);
        exercise(true, false);
        exercise(true, true);
        jumps_and_bounds();
        gated_groups();
        invalid_local_rank();
        for (const int gpu_count : {2, 4, 8}) {
            for (int local_rank = 0; local_rank < gpu_count; ++local_rank) {
                for (const bool sparse : {false, true}) {
                    cyclic_submission_order(gpu_count, local_rank, false, false, sparse);
                    cyclic_submission_order(gpu_count, local_rank, true, false, sparse);
                    cyclic_submission_order(gpu_count, local_rank, true, true, sparse);
                }
            }
        }
        std::cout << "return-copy checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
