#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace rme2 {
constexpr int gpus = 8, experts = 128, epg = 16, header_words = epg + 2;
// Header: absolute source begin, then epg+1 normalized offsets. The arrays
// remain in pinned storage; installation changes only this view and progress.
struct Installed {
    const int32_t* offsets = nullptr;
    const int32_t* indices = nullptr;
    int count = 0;
    uint64_t iteration = 0;
    std::array<int32_t, epg> heads{}, tails{};
};
inline int check_header(const int32_t* header, int capacity) {
    const auto* offsets = header + 1;
    if (header[0] < 0 || header[0] > capacity || offsets[0] != 0 ||
        offsets[epg] < 0 || offsets[epg] > capacity - header[0] ||
        !std::is_sorted(offsets, offsets + epg + 1))
        throw std::runtime_error("invalid received metadata header");
    return offsets[epg];
}
inline void install(Installed& out, const int32_t* header, const int32_t* indices,
                    int tokens, uint64_t iteration, bool validate) {
    int count = check_header(header, tokens * 8);
    const auto* offsets = header + 1;
    if (validate) {
        for (int e = 0; e < epg; ++e) {
            const auto* begin = indices + offsets[e];
            const auto* end = indices + offsets[e + 1];
            if (!std::is_sorted(begin, end) ||
                std::any_of(begin, end, [tokens](int i) { return i < 0 || i >= tokens; }))
                throw std::runtime_error("invalid received expert indices");
        }
    }
    out.offsets = offsets;
    out.indices = indices;
    out.count = count;
    out.iteration = iteration;
    out.heads.fill(0);
    out.tails.fill(0);
}
} // namespace rme2
