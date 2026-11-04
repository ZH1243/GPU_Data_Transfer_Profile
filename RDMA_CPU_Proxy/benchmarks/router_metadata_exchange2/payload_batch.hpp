#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>

namespace rme2 {
// Persistent descriptors only; no payload packing or allocation. Keep these
// alive until the receiver's completion event, and reset only next iteration.
struct PayloadBatch {
    std::array<void*, 8> destinations{}, sources{};
    std::array<std::size_t, 8> sizes{};
    std::array<int, 8> source_ranks{};
    std::size_t count = 0;
    void reset() { count = 0; }
    void append(void* dst, const void* src, std::size_t bytes, int source) {
        if (!bytes) return;
        if (!dst || !src || count == sizes.size() || source < 0 || source >= 8)
            throw std::runtime_error("invalid payload batch entry");
        destinations[count] = dst;
        sources[count] = const_cast<void*>(src); // CUDA batch API takes void**, but only reads sources.
        sizes[count] = bytes;
        source_ranks[count++] = source;
    }
};
}
