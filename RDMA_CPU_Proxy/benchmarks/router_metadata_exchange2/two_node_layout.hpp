#pragma once
#include "metadata_view.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rme2::two {
constexpr int nodes = 2, sources = 16, epg = 8;
constexpr int magic = 0x524d4532, version = 1, header_words = 16;
constexpr int directory_words = sources * header_words;
// Record: magic, version, epoch low/high, original global source, global
// destination, absolute source begin, then nine normalized expert offsets.
// A contiguous directory duplicates the small headers for one D2H submission.
inline size_t stride(int capacity) { return size_t(header_words) + capacity; }
inline size_t words(int capacity) { return directory_words + sources * stride(capacity); }
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
// All eight writes are signaled. The receive WQE carries the full epoch;
// immediate data identifies the original local source, not the relay GPU.
inline uint64_t send_id(uint64_t iteration, int source) {
    if (!iteration || iteration > (std::numeric_limits<uint64_t>::max() >> 3) ||
        source < 0 || source >= 8)
        throw std::runtime_error("invalid RDMA send identity");
    return (iteration << 3) | uint64_t(source);
}
struct CompletionProgress {
    uint64_t iteration;
    unsigned sent = 0, received = 0;
    explicit CompletionProgress(uint64_t epoch) : iteration(epoch) { send_id(epoch, 0); }
    void complete_send(uint64_t id) {
        const int source = int(id & 7);
        if (id != send_id(iteration, source) || (sent & (1u << source)))
            throw std::runtime_error("stale or duplicate RDMA send completion");
        sent |= 1u << source;
    }
    int receive_source(uint64_t wr_epoch, uint32_t source) const {
        if (wr_epoch != iteration || source >= 8 || (received & (1u << source)))
            throw std::runtime_error("stale, duplicate or invalid RDMA receive completion");
        return int(source);
    }
    void installed(int source) { received |= 1u << source; }
    bool complete() const { return sent == 255 && received == 255; }
};
}
