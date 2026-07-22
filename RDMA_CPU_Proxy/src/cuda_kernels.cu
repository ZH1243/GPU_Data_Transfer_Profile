#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

#include "config.hpp"
#include "forward_computation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace rdma_proxy {

namespace {

namespace wmma = nvcuda::wmma;

__global__ void copy_bytes_kernel(std::uint8_t* dst, const std::uint8_t* src, std::size_t bytes) {
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = tid; i < bytes; i += stride) {
        dst[i] = src[i];
    }
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ uint64_t load_acquire_system(const uint64_t* ptr) {
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(ptr)
                 : "memory");
    return value;
}

__device__ __forceinline__ void store_release_system(uint64_t* ptr, uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 :
                 : "l"(ptr), "l"(value)
                 : "memory");
}

__device__ __forceinline__ uint32_t physical_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

template <typename T>
__device__ __forceinline__ float input_to_float(T value);

template <>
__device__ __forceinline__ float input_to_float(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float input_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T output_from_float(float value);

template <>
__device__ __forceinline__ __half output_from_float(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 output_from_float(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ void stage_wmma_operands_async(
    T* a_shared,
    T* b_shared,
    const T* a,
    const T* b,
    uint32_t global_row,
    uint32_t global_column,
    uint32_t k_offset,
    uint32_t matrix_k,
    uint32_t matrix_n) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t row = lane / 2U;
    const uint32_t half_row = lane % 2U;
    const uint32_t element = half_row * 8U;
    __pipeline_memcpy_async(
        a_shared + row * 16U + element,
        a + static_cast<uint64_t>(global_row + row) * matrix_k + k_offset + element,
        16);
    __pipeline_memcpy_async(
        b_shared + row * 16U + element,
        b + static_cast<uint64_t>(k_offset + row) * matrix_n + global_column + element,
        16);
}

template <typename T>
__device__ void compute_forward_tile(const ForwardComputeTask& task, float* warp_scratch) {
    const auto* a = reinterpret_cast<const T*>(task.a_base);
    const auto* b = reinterpret_cast<const T*>(task.b_base);
    auto* d = reinterpret_cast<T*>(task.d_base);
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warps = blockDim.x / warpSize;
    const uint32_t fragment_rows = (task.valid_token_rows + 15U) / 16U;
    const uint32_t fragment_columns = (task.valid_output_columns + 15U) / 16U;
    const uint32_t fragment_count = fragment_rows * fragment_columns;
    float* fragment_output = warp_scratch + warp * 16U * 16U;
    T* operand_storage = reinterpret_cast<T*>(warp_scratch + warps * 16U * 16U);
    T* warp_operands = operand_storage + warp * 4U * 16U * 16U;
    T* a_stage[2] = {warp_operands, warp_operands + 16U * 16U};
    T* b_stage[2] = {
        warp_operands + 2U * 16U * 16U,
        warp_operands + 3U * 16U * 16U};

    for (uint32_t fragment_index = warp; fragment_index < fragment_count; fragment_index += warps) {
        const uint32_t fragment_row = fragment_index / fragment_columns;
        const uint32_t fragment_column = fragment_index % fragment_columns;
        const uint32_t local_row = fragment_row * 16U;
        const uint32_t local_column = fragment_column * 16U;
        const uint32_t global_row = task.token_row_offset + local_row;
        const uint32_t global_column = task.output_column_offset + local_column;
        const bool full_fragment =
            local_row + 16U <= task.valid_token_rows &&
            local_column + 16U <= task.valid_output_columns;

        if (full_fragment) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0F);
            stage_wmma_operands_async(
                a_stage[0], b_stage[0], a, b, global_row, global_column, 0, task.matrix_k, task.matrix_n);
            __pipeline_commit();
            const uint32_t k_tiles = task.matrix_k / 16U;
            for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                const uint32_t stage = k_tile & 1U;
                const bool has_next = k_tile + 1U < k_tiles;
                if (has_next) {
                    const uint32_t next_stage = stage ^ 1U;
                    stage_wmma_operands_async(
                        a_stage[next_stage],
                        b_stage[next_stage],
                        a,
                        b,
                        global_row,
                        global_column,
                        (k_tile + 1U) * 16U,
                        task.matrix_k,
                        task.matrix_n);
                    __pipeline_commit();
                    __pipeline_wait_prior(1);
                } else {
                    __pipeline_wait_prior(0);
                }
                __syncwarp();
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_fragment;
                wmma::load_matrix_sync(a_fragment, a_stage[stage], 16);
                wmma::load_matrix_sync(b_fragment, b_stage[stage], 16);
                wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
                __syncwarp();
            }
            wmma::store_matrix_sync(fragment_output, accumulator, 16, wmma::mem_row_major);
            __syncwarp();
            for (uint32_t element = lane; element < 256U; element += warpSize) {
                const uint32_t row = element / 16U;
                const uint32_t column = element % 16U;
                d[static_cast<uint64_t>(global_row + row) * task.matrix_n + global_column + column] =
                    output_from_float<T>(fragment_output[element]);
            }
            __syncwarp();
            continue;
        }

        // Edge fragments are uncommon for the intended 128x128 tiles. A
        // scalar fallback keeps partial notifications and output tails correct.
        for (uint32_t element = lane; element < 256U; element += warpSize) {
            const uint32_t row = element / 16U;
            const uint32_t column = element % 16U;
            if (local_row + row >= task.valid_token_rows ||
                local_column + column >= task.valid_output_columns) {
                continue;
            }
            float sum = 0.0F;
            for (uint32_t k = 0; k < task.matrix_k; ++k) {
                sum += input_to_float(a[static_cast<uint64_t>(global_row + row) * task.matrix_k + k]) *
                       input_to_float(b[static_cast<uint64_t>(k) * task.matrix_n + global_column + column]);
            }
            d[static_cast<uint64_t>(global_row + row) * task.matrix_n + global_column + column] =
                output_from_float<T>(sum);
        }
    }
}

template <typename T>
__device__ __forceinline__ void consume_staged_operands(const T* a_shared, const T* b_shared) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t stage_element = (lane / 2U) * 16U + (lane % 2U) * 8U;
    const volatile uint16_t* a_bits = reinterpret_cast<const volatile uint16_t*>(a_shared);
    const volatile uint16_t* b_bits = reinterpret_cast<const volatile uint16_t*>(b_shared);
    const uint32_t sink = static_cast<uint32_t>(a_bits[stage_element]) |
        (static_cast<uint32_t>(b_bits[stage_element]) << 16U);
    // The empty asm makes the shared-memory reads observable to the compiler,
    // preventing the global-to-shared copies from being optimized away.
    asm volatile("" : : "r"(sink) : "memory");
}

template <typename T>
__device__ __forceinline__ void stage_partial_operands(
    T* a_shared,
    T* b_shared,
    const T* a,
    const T* b,
    uint32_t global_row,
    uint32_t global_column,
    uint32_t local_row,
    uint32_t local_column,
    uint32_t k_offset,
    const ForwardComputeTask& task) {
    const uint32_t lane = threadIdx.x % warpSize;
    for (uint32_t element = lane; element < 256U; element += warpSize) {
        const uint32_t row = element / 16U;
        const uint32_t column = element % 16U;
        a_shared[element] = local_row + row < task.valid_token_rows
            ? a[static_cast<uint64_t>(global_row + row) * task.matrix_k + k_offset + column]
            : T{};
        b_shared[element] = local_column + column < task.valid_output_columns
            ? b[static_cast<uint64_t>(k_offset + row) * task.matrix_n + global_column + column]
            : T{};
    }
}

template <typename T>
__device__ void load_forward_tile_operands(const ForwardComputeTask& task, float* warp_scratch) {
    const auto* a = reinterpret_cast<const T*>(task.a_base);
    const auto* b = reinterpret_cast<const T*>(task.b_base);
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t warps = blockDim.x / warpSize;
    const uint32_t fragment_rows = (task.valid_token_rows + 15U) / 16U;
    const uint32_t fragment_columns = (task.valid_output_columns + 15U) / 16U;
    const uint32_t fragment_count = fragment_rows * fragment_columns;
    T* operand_storage = reinterpret_cast<T*>(warp_scratch + warps * 16U * 16U);
    T* warp_operands = operand_storage + warp * 4U * 16U * 16U;
    T* a_stage[2] = {warp_operands, warp_operands + 16U * 16U};
    T* b_stage[2] = {
        warp_operands + 2U * 16U * 16U,
        warp_operands + 3U * 16U * 16U};

    for (uint32_t fragment_index = warp; fragment_index < fragment_count; fragment_index += warps) {
        const uint32_t fragment_row = fragment_index / fragment_columns;
        const uint32_t fragment_column = fragment_index % fragment_columns;
        const uint32_t local_row = fragment_row * 16U;
        const uint32_t local_column = fragment_column * 16U;
        const uint32_t global_row = task.token_row_offset + local_row;
        const uint32_t global_column = task.output_column_offset + local_column;
        const bool full_fragment =
            local_row + 16U <= task.valid_token_rows &&
            local_column + 16U <= task.valid_output_columns;
        const uint32_t k_tiles = task.matrix_k / 16U;

        if (full_fragment) {
            stage_wmma_operands_async(
                a_stage[0], b_stage[0], a, b, global_row, global_column, 0, task.matrix_k, task.matrix_n);
            __pipeline_commit();
            for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                const uint32_t stage = k_tile & 1U;
                const bool has_next = k_tile + 1U < k_tiles;
                if (has_next) {
                    const uint32_t next_stage = stage ^ 1U;
                    stage_wmma_operands_async(
                        a_stage[next_stage],
                        b_stage[next_stage],
                        a,
                        b,
                        global_row,
                        global_column,
                        (k_tile + 1U) * 16U,
                        task.matrix_k,
                        task.matrix_n);
                    __pipeline_commit();
                    __pipeline_wait_prior(1);
                } else {
                    __pipeline_wait_prior(0);
                }
                __syncwarp();
                consume_staged_operands(a_stage[stage], b_stage[stage]);
                __syncwarp();
            }
            continue;
        }

        // Partial row/output fragments use guarded scalar loads into shared
        // memory so a tail task never reads beyond A or B.
        for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
            const uint32_t stage = k_tile & 1U;
            stage_partial_operands(
                a_stage[stage],
                b_stage[stage],
                a,
                b,
                global_row,
                global_column,
                local_row,
                local_column,
                k_tile * 16U,
                task);
            __syncwarp();
            consume_staged_operands(a_stage[stage], b_stage[stage]);
            __syncwarp();
        }
    }
}

template <typename T, bool LoadOnly, bool DequeueOnly>
__global__ void persistent_forward_computation_kernel(
    const ForwardDeviceQueueView* queue_views,
    uint32_t num_queues,
    ForwardQueueSignal* abort_signal,
    uint32_t* physical_sm_ids) {
    extern __shared__ __align__(32) unsigned char shared_storage_bytes[];
    auto* shared_storage = reinterpret_cast<float*>(shared_storage_bytes);
    __shared__ ForwardComputeTask shared_task;
    __shared__ uint64_t shared_position;
    __shared__ int shared_state;
    const uint32_t queue_index = blockIdx.x % num_queues;
    const ForwardDeviceQueueView queue = queue_views[queue_index];
    uint64_t local_empty_polls = 0;

    if (threadIdx.x == 0) {
        physical_sm_ids[blockIdx.x] = physical_smid();
        // Touch the reserved tail so the one-CTA-per-SM allocation cannot be
        // optimized away. The launch reserves > half of per-SM shared memory.
        reinterpret_cast<volatile uint8_t*>(shared_storage)[0] = 0;
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            shared_state = 0;
            if (load_acquire_system(&abort_signal->sequence) != 0) {
                shared_state = -1;
            } else {
                const uint64_t position = atomicAdd(
                    reinterpret_cast<unsigned long long*>(queue.dequeue_position), 0ULL);
                // published_head is device memory. Keeping this hot poll target in
                // HBM avoids repeated mapped-host reads over PCIe.
                const uint64_t published_head = load_acquire_system(queue.published_head);
                if (position < published_head) {
                    const auto prior = atomicCAS(
                        reinterpret_cast<unsigned long long*>(queue.dequeue_position),
                        static_cast<unsigned long long>(position),
                        static_cast<unsigned long long>(position + 1));
                    if (prior == position) {
                        shared_task = queue.tasks[position % queue.capacity];
                        shared_position = position;
                        shared_state = 1;
                    }
                }
            }
        }
        __syncthreads();
        if (shared_state < 0) return;
        if (shared_state == 0) {
            ++local_empty_polls;
            __nanosleep(64);
            __syncthreads();
            continue;
        }

        // The descriptor is now private to this CTA in shared memory. Commit
        // dequeues in logical-position order before publishing the compact
        // host-visible tail. Claiming can occur out of order across CTAs, so a
        // separate device-local commit position prevents a later claim from
        // making an earlier, not-yet-copied descriptor appear reusable.
        if (threadIdx.x == 0) {
            while (load_acquire_system(queue.dequeue_commit_position) != shared_position) {
                __nanosleep(64);
            }
            __threadfence_system();
            store_release_system(&queue.dequeued_tail->sequence, shared_position + 1);
            __threadfence_system();
            store_release_system(queue.dequeue_commit_position, shared_position + 1);
        }
        __syncthreads();

        if (threadIdx.x == 0 && local_empty_polls != 0) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(&queue.stats->poll_iterations),
                static_cast<unsigned long long>(local_empty_polls));
            local_empty_polls = 0;
        }
        __syncthreads();

        bool valid = true;
        if (shared_task.generation != queue.generation) {
            if (threadIdx.x == 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&queue.stats->stale_tasks), 1ULL);
            }
            valid = false;
        } else if (shared_task.type == static_cast<uint32_t>(ForwardTaskType::kExit)) {
            if (threadIdx.x == 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&queue.stats->exit_tasks_consumed), 1ULL);
            }
            return;
        } else if (
            shared_task.type != static_cast<uint32_t>(ForwardTaskType::kCompute) ||
            shared_task.dtype != static_cast<uint32_t>(std::is_same<T, __nv_bfloat16>::value
                                                            ? DataType::kBF16
                                                            : DataType::kFP16) ||
            shared_task.a_base == 0 || shared_task.b_base == 0 || shared_task.d_base == 0 ||
            shared_task.matrix_k == 0 || shared_task.matrix_k % 16U != 0 ||
            shared_task.matrix_n == 0 || shared_task.matrix_n % 16U != 0 ||
            shared_task.valid_token_rows == 0 || shared_task.valid_output_columns == 0) {
            if (threadIdx.x == 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&queue.stats->invalid_tasks), 1ULL);
            }
            valid = false;
        }

        if (valid) {
            if (threadIdx.x == 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&queue.stats->tasks_claimed), 1ULL);
            }
            if constexpr (DequeueOnly) {
                // Queue benchmarking mode: claiming and acknowledging the task
                // is the entire payload. In particular, do not dereference A,
                // B, or D and do not stage any tensor data in shared memory.
            } else if constexpr (LoadOnly) {
                load_forward_tile_operands<T>(shared_task, shared_storage);
            } else {
                compute_forward_tile<T>(shared_task, shared_storage);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&queue.stats->tasks_completed), 1ULL);
            }
        }
        __syncthreads();
    }
}

template <typename T, bool LoadOnly, bool DequeueOnly>
void launch_persistent_typed(
    const ForwardDeviceQueueView* queues,
    uint32_t num_queues,
    uint32_t num_ctas,
    ForwardQueueSignal* abort_signal,
    uint32_t* physical_sm_ids,
    std::size_t dynamic_shared_bytes,
    cudaStream_t stream) {
    check_cuda(
        cudaFuncSetAttribute(
            persistent_forward_computation_kernel<T, LoadOnly, DequeueOnly>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dynamic_shared_bytes)),
        "cudaFuncSetAttribute persistent dynamic shared memory");
    check_cuda(
        cudaFuncSetAttribute(
            persistent_forward_computation_kernel<T, LoadOnly, DequeueOnly>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute persistent shared-memory carveout");
    int active_blocks_per_sm = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            persistent_forward_computation_kernel<T, LoadOnly, DequeueOnly>,
            256,
            dynamic_shared_bytes),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor persistent computation");
    if (active_blocks_per_sm != 1) {
        throw std::runtime_error(
            "persistent forwarding computation requires exactly one resident CTA per SM; occupancy API reported " +
            std::to_string(active_blocks_per_sm));
    }
    persistent_forward_computation_kernel<T, LoadOnly, DequeueOnly>
        <<<num_ctas, 256, dynamic_shared_bytes, stream>>>(
        queues, num_queues, abort_signal, physical_sm_ids);
    check_cuda(cudaGetLastError(), "persistent forwarding computation kernel launch");
}

}  // namespace

void launch_copy_tokens_kernel(void* dst, const void* src, std::size_t bytes) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((bytes + threads - 1) / threads);
    copy_bytes_kernel<<<std::max(1, blocks), threads>>>(
        static_cast<std::uint8_t*>(dst),
        static_cast<const std::uint8_t*>(src),
        bytes);
    check_cuda(cudaGetLastError(), "copy_bytes_kernel launch");
}

void launch_persistent_forward_computation_kernel(
    const ForwardDeviceQueueView* queues,
    uint32_t num_queues,
    uint32_t num_ctas,
    DataType dtype,
    bool load_only,
    bool dequeue_only,
    ForwardQueueSignal* abort_signal,
    uint32_t* physical_sm_ids,
    std::size_t dynamic_shared_bytes,
    void* stream) {
    if (!queues || !abort_signal || !physical_sm_ids || num_queues == 0 || num_ctas == 0) {
        throw std::runtime_error("invalid persistent forwarding computation kernel launch parameters");
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (dtype == DataType::kBF16) {
        if (dequeue_only) {
            launch_persistent_typed<__nv_bfloat16, false, true>(
                queues, num_queues, num_ctas, abort_signal, physical_sm_ids, dynamic_shared_bytes, cuda_stream);
        } else if (load_only) {
            launch_persistent_typed<__nv_bfloat16, true, false>(
                queues, num_queues, num_ctas, abort_signal, physical_sm_ids, dynamic_shared_bytes, cuda_stream);
        } else {
            launch_persistent_typed<__nv_bfloat16, false, false>(
                queues, num_queues, num_ctas, abort_signal, physical_sm_ids, dynamic_shared_bytes, cuda_stream);
        }
    } else if (dtype == DataType::kFP16) {
        if (dequeue_only) {
            launch_persistent_typed<__half, false, true>(
                queues, num_queues, num_ctas, abort_signal, physical_sm_ids, dynamic_shared_bytes, cuda_stream);
        } else if (load_only) {
            launch_persistent_typed<__half, true, false>(
                queues, num_queues, num_ctas, abort_signal, physical_sm_ids, dynamic_shared_bytes, cuda_stream);
        } else {
            launch_persistent_typed<__half, false, false>(
                queues, num_queues, num_ctas, abort_signal, physical_sm_ids, dynamic_shared_bytes, cuda_stream);
        }
    } else {
        throw std::runtime_error("persistent forwarding computation supports only BF16/FP16");
    }
}

}  // namespace rdma_proxy
