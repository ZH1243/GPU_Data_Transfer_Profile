#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

constexpr int max_ranks = 32;
inline size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) throw std::runtime_error("size overflow");
    return a + b;
}
inline size_t checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) throw std::runtime_error("size overflow");
    return a * b;
}
inline size_t number(const std::string& s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("expected nonnegative integer: " + s);
    auto n = std::stoull(s);
    if (n > std::numeric_limits<size_t>::max()) throw std::runtime_error("integer overflow");
    return static_cast<size_t>(n);
}
inline std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        auto end = s.find(',', start);
        auto token = s.substr(start, end - start);
        if (token.empty()) throw std::runtime_error("empty list element");
        out.push_back(token);
        if (end == std::string::npos) return out;
        start = end + 1;
    }
}
inline size_t bytes(const std::string& s) {
    auto end = s.find_first_not_of("0123456789");
    auto n = number(s.substr(0, end));
    auto unit = end == std::string::npos ? "" : s.substr(end);
    size_t scale = 1;
    if (unit == "KB") scale = 1000;
    else if (unit == "MB") scale = 1000000;
    else if (unit == "GB") scale = 1000000000;
    else if (unit == "KiB") scale = 1024;
    else if (unit == "MiB") scale = 1024 * 1024;
    else if (unit == "GiB") scale = 1024ULL * 1024 * 1024;
    else if (unit != "" && unit != "B") throw std::runtime_error("unknown size unit: " + unit);
    if (!n) throw std::runtime_error("copy sizes must be positive");
    return checked_mul(n, scale);
}
struct Layout {
    std::vector<size_t> sizes, prefix;
    size_t batches, batch_bytes = 0, source_bytes;
    Layout(std::vector<size_t> s, size_t b) : sizes(std::move(s)), batches(b) {
        if (!b || sizes.empty()) throw std::runtime_error("empty layout");
        for (auto n : sizes) {
            if (!n) throw std::runtime_error("zero copy size");
            prefix.push_back(batch_bytes);
            batch_bytes = checked_add(batch_bytes, n);
        }
        source_bytes = checked_mul(batch_bytes, batches);
    }
    size_t src(size_t batch, size_t step) const { return batch * batch_bytes + prefix.at(step - 1); }
    size_t dst(size_t batch, size_t step) const { return batch * sizes.at(step - 1); }
    size_t receive(size_t step) const { return checked_mul(batches, sizes.at(step - 1)); }
};
// GCC/Clang lock-free atomics on coherent MAP_SHARED memory, not volatile polling.
struct alignas(128) Flag { uint64_t value = 0; };
struct BarrierState { Flag epochs[max_ranks]; alignas(128) uint32_t abort = 0; };
static_assert(__atomic_always_lock_free(8, nullptr), "64-bit atomics must be lock-free");
static_assert(__atomic_always_lock_free(4, nullptr), "32-bit atomics must be lock-free");
inline void abort_all(BarrierState& s) { __atomic_store_n(&s.abort, 1U, __ATOMIC_RELEASE); }
inline bool aborted(BarrierState& s) { return __atomic_load_n(&s.abort, __ATOMIC_ACQUIRE) != 0; }
inline void relax_cpu() {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}
class Barrier {
    BarrierState& state;
    int rank, n;
    uint64_t epoch = 0;
    std::chrono::seconds timeout;
public:
    Barrier(BarrierState& s, int r, int count, unsigned seconds)
        : state(s), rank(r), n(count), timeout(seconds) {}
    void wait() {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        __atomic_store_n(&state.epochs[rank].value, ++epoch, __ATOMIC_RELEASE);
        unsigned spins = 0;
        for (int peer = 0; peer < n; ++peer) {
            // >= matters: a fast peer may already have entered the next barrier.
            while (__atomic_load_n(&state.epochs[peer].value, __ATOMIC_ACQUIRE) < epoch) {
                relax_cpu();
                if ((++spins & 4095U) == 0) {
                    if (aborted(state)) throw std::runtime_error("another proxy failed");
                    if (std::chrono::steady_clock::now() > deadline) {
                        abort_all(state);
                        throw std::runtime_error("barrier timeout at epoch " + std::to_string(epoch));
                    }
                }
            }
        }
        if (aborted(state)) throw std::runtime_error("another proxy failed");
    }
};
