#include "exchange.cuh"
#include "metadata_view.hpp"

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
