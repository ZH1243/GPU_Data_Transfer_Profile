#pragma once
#include "metadata_view.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rme2::two {
constexpr int nodes = 2, sources = 16, epg = 8;
constexpr int magic = 0x524d4532, version = 2, header_words = 16;
constexpr int x3_x4_magic = 0x58335834, x3_x4_tag = 8, messages = 9;
constexpr int directory_words = (sources + 1) * header_words;
// Record: magic, version, epoch low/high, original global source, global
// destination, absolute source begin, then nine normalized expert offsets.
// A contiguous directory duplicates the small headers for one D2H submission.
inline size_t stride(int capacity) { return size_t(header_words) + capacity; }
// One additional same-index source record: 64-byte header, count int32 x3
// entries, then count uint8 x4 masks. Only the valid prefix goes over RDMA.
inline size_t x3_x4_offset(int capacity) { return directory_words + sources * stride(capacity); }
inline size_t x3_x4_bytes(int count) { return header_words * sizeof(int32_t) + size_t(count) * 5; }
inline size_t words(int capacity) { return x3_x4_offset(capacity) + (x3_x4_bytes(capacity / 8) + 3) / 4; }
inline size_t record_offset(int slot, int capacity) {
    return directory_words + size_t(slot) * stride(capacity);
}
inline uint64_t epoch(const int32_t* h) {
    return uint64_t(uint32_t(h[2])) | (uint64_t(uint32_t(h[3])) << 32);
}
inline int check_record(const int32_t* h, int capacity, uint64_t iteration,
                        int source, int destination) {
    if (h[0] != magic || h[1] != version || epoch(h) != iteration ||
        h[4] != source || h[5] != destination)
        throw std::runtime_error("two-node metadata identity/epoch mismatch");
    return rme2::check_header(h + 6, capacity, epg);
}
inline size_t record_bytes(int count) { return (size_t(header_words) + count) * sizeof(int32_t); }
inline void check_received(const int32_t* h, int capacity, uint64_t iteration,
                           int source, int destination, size_t bytes) {
    if (bytes < record_bytes(0) || bytes > record_bytes(capacity))
        throw std::runtime_error("invalid RDMA record length");
    if (record_bytes(check_record(h, capacity, iteration, source, destination)) != bytes)
        throw std::runtime_error("RDMA record length/count mismatch");
}
// x3/x4 header: magic, version, epoch low/high, source, destination, count,
// followed by nine reserved zero words. Source/destination must be same-index
// proxies on opposite nodes. The directory stores our outgoing header; the
// pinned record itself receives the remote source's data directly from RDMA.
inline int check_x3_x4_header(const int32_t* h, int tokens, uint64_t iteration,
                              int source, int destination) {
    if (h[0] != x3_x4_magic || h[1] != version || epoch(h) != iteration ||
        source < 0 || source >= sources || destination != (source ^ 8) ||
        h[4] != source || h[5] != destination || h[6] < 0 || h[6] > tokens ||
        std::any_of(h + 7, h + header_words, [](int32_t value) { return value != 0; }))
        throw std::runtime_error("invalid x3/x4 identity, epoch or length");
    return h[6];
}
struct InstalledX3X4 {
    const int32_t* x3 = nullptr;
    const uint8_t* x4 = nullptr;
    int count = 0;
    uint64_t iteration = 0;
};
inline void install_x3_x4(InstalledX3X4& out, const int32_t* h, int tokens,
                          uint64_t iteration, int source, int destination,
                          size_t bytes, bool validate) {
    if (bytes < x3_x4_bytes(0) || bytes > x3_x4_bytes(tokens))
        throw std::runtime_error("invalid x3/x4 RDMA record length");
    const int count = check_x3_x4_header(h, tokens, iteration, source, destination);
    if (bytes != x3_x4_bytes(count))
        throw std::runtime_error("x3/x4 RDMA record length/count mismatch");
    const auto* x3 = h + header_words;
    const auto* x4 = reinterpret_cast<const uint8_t*>(x3 + count);
    if (validate) {
        for (int i = 0; i < count; ++i) {
            if (x3[i] < 0 || x3[i] >= tokens || x4[i] == 0 ||
                (i && (x4[i] > x4[i - 1] || (x4[i] == x4[i - 1] && x3[i] <= x3[i - 1]))))
                throw std::runtime_error("invalid x3/x4 token, mask or stable ordering");
        }
    }
    out = {x3, x4, count, iteration};
}
// All nine writes are signaled. Immediate tags 0..7 identify expert sources;
// tag 8 identifies the same-index x3/x4 record. Receive WQEs carry the epoch.
inline uint64_t send_id(uint64_t iteration, int source) {
    if (!iteration || iteration > (std::numeric_limits<uint64_t>::max() >> 4) ||
        source < 0 || source >= messages)
        throw std::runtime_error("invalid RDMA send identity");
    return (iteration << 4) | uint64_t(source);
}
struct CompletionProgress {
    uint64_t iteration;
    unsigned sent = 0, received = 0;
    explicit CompletionProgress(uint64_t epoch) : iteration(epoch) { send_id(epoch, 0); }
    void complete_send(uint64_t id) {
        const int source = int(id & 15);
        if (id != send_id(iteration, source) || (sent & (1u << source)))
            throw std::runtime_error("stale or duplicate RDMA send completion");
        sent |= 1u << source;
    }
    int receive_source(uint64_t wr_epoch, uint32_t source) const {
        if (wr_epoch != iteration || source >= messages || (received & (1u << source)))
            throw std::runtime_error("stale, duplicate or invalid RDMA receive completion");
        return int(source);
    }
    void installed(int source) { received |= 1u << source; }
    bool complete() const { return sent == (1u << messages) - 1 && received == (1u << messages) - 1; }
};
}
