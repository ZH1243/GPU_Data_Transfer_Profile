#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
struct PeerArenas { int32_t* ptr[8]; };
void launch_exchange(PeerArenas peers, int source, int capacity,
                     const int32_t* offsets, const int32_t* values, cudaStream_t stream);
