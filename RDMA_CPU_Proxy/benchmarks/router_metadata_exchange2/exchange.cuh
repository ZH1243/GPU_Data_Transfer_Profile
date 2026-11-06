#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
struct PeerArenas { int32_t* ptr[8]; };
void launch_exchange(PeerArenas peers, int source, int capacity,
                     const int32_t* offsets, const int32_t* values, cudaStream_t stream);
void launch_exchange_two(PeerArenas peers, int source, int node, int capacity,
                         uint64_t iteration, const int32_t* offsets,
                         const int32_t* values, const int32_t* node_offsets,
                         const int32_t* x3, const uint8_t* x4, cudaStream_t stream);
