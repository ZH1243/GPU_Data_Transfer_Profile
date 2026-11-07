#pragma once
#include "two_node_layout.hpp"

namespace rme2::receiver {
// Outside the registered wire arena. Each source owns one cache-line-separated
// readiness flag on each local destination GPU; epochs are never reset/reused.
struct alignas(64) Flag { uint64_t epoch; };
struct alignas(64) Result { uint64_t epoch; int error; };
constexpr int timeout_error = 1, record_error = 2;
inline size_t flags_offset(int capacity) {
    const size_t bytes = two::words(capacity) * sizeof(int32_t);
    return (bytes + 63) & ~size_t(63);
}
inline size_t allocation_bytes(int capacity) {
    return flags_offset(capacity) + 8 * sizeof(Flag);
}
} // namespace rme2::receiver
