#include "exchange.cuh"
#include "metadata_view.hpp"
#include "two_node_layout.hpp"

// One CTA per destination; no intermediate packing allocation. Self only
// writes its small header: its D2H payload is read directly from CuTe output.
__global__ void exchange_kernel(PeerArenas peers, int source, int capacity,
                                const int32_t* offsets, const int32_t* values) {
    const int dest = blockIdx.x;
    const int begin = offsets[dest * rme2::epg];
    const int end = offsets[(dest + 1) * rme2::epg];
    int32_t* header = peers.ptr[dest] + source * rme2::header_words;
    if (begin < 0 || end < begin || end > capacity) {
        if (threadIdx.x == 0) header[0] = -1;
        __threadfence_system();
        return;
    }
    if (threadIdx.x == 0) header[0] = begin;
    if (threadIdx.x <= rme2::epg)
        header[1 + threadIdx.x] = offsets[dest * rme2::epg + threadIdx.x] - begin;
    if (dest != source) {
        int32_t* dst = peers.ptr[dest] + rme2::gpus * rme2::header_words
                       + size_t(source) * capacity;
        for (int i = threadIdx.x; i < end - begin; i += blockDim.x)
            dst[i] = values[begin + i];
    }
    // Every writer fences its own writes before the stream completion event.
    __threadfence_system();
}

void launch_exchange(PeerArenas peers, int source, int capacity,
                     const int32_t* offsets, const int32_t* values, cudaStream_t stream) {
    exchange_kernel<<<rme2::gpus, 256, 0, stream>>>(peers, source, capacity, offsets, values);
}

// One CTA per global destination. Each local relay owns two banks of eight
// original-source records. Self payloads are copied too, so all RDMA sources
// belong to the relay's registered allocation (never an imported IPC pointer).
__global__ void exchange_two_kernel(PeerArenas peers, int source, int node, int capacity,
                                    uint64_t iteration, const int32_t* offsets,
                                    const int32_t* values) {
    const int dest = blockIdx.x, relay = dest % 8, slot = (dest / 8) * 8 + source;
    int32_t* arena = peers.ptr[relay];
    int32_t* record = arena + rme2::two::directory_words +
                      size_t(slot) * (size_t(capacity) + rme2::two::header_words);
    int32_t* directory = arena + slot * rme2::two::header_words;
    const int begin = offsets[dest * 8], end = offsets[(dest + 1) * 8];
    if (threadIdx.x < rme2::two::header_words) {
        const int t = threadIdx.x;
        int32_t value = 0;
        if (t == 0) value = rme2::two::magic;
        else if (t == 1) value = rme2::two::version;
        else if (t == 2) value = int32_t(uint32_t(iteration));
        else if (t == 3) value = int32_t(uint32_t(iteration >> 32));
        else if (t == 4) value = node * 8 + source;
        else if (t == 5) value = dest;
        else if (t == 6) value = begin;
        else value = offsets[dest * 8 + t - 7] - begin;
        if (begin < 0 || end < begin || end > capacity) value = -1;
        directory[t] = record[t] = value;
    }
    if (begin >= 0 && end >= begin && end <= capacity)
        for (int i = threadIdx.x; i < end - begin; i += blockDim.x)
            record[rme2::two::header_words + i] = values[begin + i];
    __threadfence_system();
}
void launch_exchange_two(PeerArenas peers, int source, int node, int capacity,
                         uint64_t iteration, const int32_t* offsets,
                         const int32_t* values, cudaStream_t stream) {
    exchange_two_kernel<<<16, 256, 0, stream>>>(peers, source, node, capacity, iteration, offsets, values);
}
