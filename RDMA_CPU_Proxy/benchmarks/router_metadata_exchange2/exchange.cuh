#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include "receiver_layout.hpp"
struct PeerArenas { int32_t* ptr[8]; };
void launch_exchange(PeerArenas peers, int source, int capacity,
                     const int32_t* offsets, const int32_t* values, cudaStream_t stream);
void launch_exchange_two(PeerArenas peers, int source, int node, int capacity,
                         uint64_t iteration, const int32_t* offsets,
                         const int32_t* values, const int32_t* node_offsets,
                         const int32_t* x3, const uint8_t* x4, bool direct_x3_x4_output,
                         bool receiver_kernel,
                         cudaStream_t stream);
// Seven independent CTAs consume only this node's peer-source expert records.
// Pinned output/results must be GPU-mapped. CPU consumption requires completion.
void launch_receiver(int32_t* inbox, int32_t* mapped_host, int rank, int node,
                     int capacity, uint64_t iteration, uint64_t timeout_cycles,
                     rme2::receiver::Result* results, cudaStream_t stream);
