#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define RDMA_PROXY_C_API __declspec(dllexport)
#elif defined(__GNUC__)
#define RDMA_PROXY_C_API __attribute__((visibility("default")))
#else
#define RDMA_PROXY_C_API
#endif

/*
 * Stable C ABI for embedding the C++ proxy in another process (for example,
 * a Python worker using ctypes). The handle is intentionally opaque so C++
 * types and standard-library objects never cross the shared-library boundary.
 */
typedef struct RdmaProxyHandle RdmaProxyHandle;

/* Device-buffer kinds requested through RdmaProxyDeviceBufferAllocator. */
typedef enum RdmaProxyDeviceBufferKind {
    RDMA_PROXY_DEVICE_BUFFER_RDMA_SEND = 1,
    RDMA_PROXY_DEVICE_BUFFER_RDMA_RECEIVE = 2,
    RDMA_PROXY_DEVICE_BUFFER_NVLINK_RECEIVE = 3,
    RDMA_PROXY_DEVICE_BUFFER_ROUTER_A_IDX = 4,
    RDMA_PROXY_DEVICE_BUFFER_GATHER_TABLE = 5,
    RDMA_PROXY_DEVICE_BUFFER_GATHER_READY_ROWS = 6,
} RdmaProxyDeviceBufferKind;

typedef enum RdmaProxyDeviceBufferElementType {
    RDMA_PROXY_DEVICE_BUFFER_BF16 = 1,
    RDMA_PROXY_DEVICE_BUFFER_FP16 = 2,
    RDMA_PROXY_DEVICE_BUFFER_FP32 = 3,
    RDMA_PROXY_DEVICE_BUFFER_INT32 = 4,
} RdmaProxyDeviceBufferElementType;

/*
 * Allocation request passed to an embedding runtime. dimensions contains up
 * to two logical tensor dimensions; bytes is always the required allocation
 * size. Negative peer/source identifiers mean "not applicable". A router
 * NVLINK_RECEIVE request whose remote source GPU equals this proxy's local GPU
 * is a logical alias request: the allocator must return the pointer previously
 * supplied for that source node's RDMA_RECEIVE request.
 */
typedef struct RdmaProxyDeviceBufferRequest {
    uint32_t struct_size;
    uint32_t version;
    int32_t kind;
    int32_t element_type;
    int32_t peer_rank;
    int32_t source_node_rank;
    int32_t source_gpu_index;
    uint32_t dimension_count;
    uint64_t bytes;
    uint64_t dimensions[2];
} RdmaProxyDeviceBufferRequest;

/* Return a device pointer as uintptr_t, or zero to reject the request. */
typedef uintptr_t (*RdmaProxyDeviceBufferAllocator)(
    void* context,
    const RdmaProxyDeviceBufferRequest* request);

/* Increment this value only when the exported ABI changes incompatibly. */
RDMA_PROXY_C_API int rdma_proxy_abi_version(void);

/*
 * Create a proxy from the same argv accepted by the rdma_cpu_proxy executable.
 * The argv strings are copied before this function returns.
 * Returns NULL on failure; retrieve the diagnostic with rdma_proxy_last_error.
 */
RDMA_PROXY_C_API RdmaProxyHandle* rdma_proxy_create(
    int argc,
    const char* const* argv);

/*
 * Install an allocator before rdma_proxy_initialize(). The proxy borrows all
 * returned buffers and never frees them; the embedding runtime must keep them
 * alive until after rdma_proxy_shutdown()/rdma_proxy_destroy(). Passing NULL
 * restores the proxy's normal internal CUDA allocation behavior.
 */
RDMA_PROXY_C_API int rdma_proxy_set_device_buffer_allocator(
    RdmaProxyHandle* handle,
    RdmaProxyDeviceBufferAllocator allocator,
    void* context);

RDMA_PROXY_C_API int rdma_proxy_initialize(RdmaProxyHandle* handle);
RDMA_PROXY_C_API int rdma_proxy_run(RdmaProxyHandle* handle);

/* Finite-run iteration control for embedding runtimes that own the main loop. */
RDMA_PROXY_C_API int rdma_proxy_get_num_iterations(
    RdmaProxyHandle* handle,
    uint64_t* num_iterations);
/*
 * Reset the gather-table publication state for one iteration and wait until
 * ready_rows=0 is visible on the configured GPU. This lets an embedding
 * runtime safely launch a readiness-gated persistent kernel before calling
 * rdma_proxy_run_iteration, possibly from another host thread.
 */
RDMA_PROXY_C_API int rdma_proxy_prepare_iteration(
    RdmaProxyHandle* handle,
    uint64_t iteration);
RDMA_PROXY_C_API int rdma_proxy_run_iteration(
    RdmaProxyHandle* handle,
    uint64_t iteration);

/* GPU-event timing for readiness-gated computation launched by the embedder. */
RDMA_PROXY_C_API int rdma_proxy_record_computation_end(
    RdmaProxyHandle* handle,
    uint64_t iteration,
    uintptr_t cuda_stream);
RDMA_PROXY_C_API int rdma_proxy_get_computation_elapsed_ms(
    RdmaProxyHandle* handle,
    uint64_t iteration,
    float* elapsed_ms);
RDMA_PROXY_C_API int rdma_proxy_get_computation_num_tokens(
    RdmaProxyHandle* handle,
    uint64_t* num_tokens);
/* Query length with counts=NULL, capacity=0; otherwise copy all local expert
 * counts in scheduler expert order. Counts exclude A_idx capacity padding. */
RDMA_PROXY_C_API int rdma_proxy_get_computation_expert_counts(
    RdmaProxyHandle* handle, uint64_t* counts, uint64_t capacity,
    uint64_t* num_experts);
RDMA_PROXY_C_API int rdma_proxy_finish(RdmaProxyHandle* handle);

/* Configure once after initialization, before the first prepare_iteration.
 * The proxy owns the mapped flag. Finish GPU writes before shutdown/destroy.
 * prepare_iteration resets/arms it; the notification receiver polls it.
 * Times are CPU detection microseconds since iteration preparation, including
 * up computation and host launch/compilation delays, not GPU timestamps.
 * read waits for the final prefix; capacity must hold up to final_groups entries.
 */
RDMA_PROXY_C_API int rdma_proxy_configure_down_ready_trace(
    RdmaProxyHandle* handle, int32_t final_groups, uintptr_t* device_address);
RDMA_PROXY_C_API int rdma_proxy_read_down_ready_trace(
    RdmaProxyHandle* handle, uint64_t iteration, int32_t* groups,
    double* times_us, uint64_t capacity, uint64_t* count);

typedef struct RdmaProxyReturnCopy {
    uintptr_t source;
    uintptr_t destination;
    uint64_t capacity_bytes; /* Minimum of source and destination capacities. */
    uint64_t row_index;      /* Index into host_valid_rows. */
    int32_t peer;            /* Same-node destination GPU index; batches group by peer. */
} RdmaProxyReturnCopy;

/* Configure after down-ready trace, before the first iteration. Pointers/events
 * are borrowed until shutdown. host_valid_rows must be pinned host int64 storage,
 * updated on the compute stream BEFORE down GEMM. tile_n=0 means row-major.
 * All destination buffers use the same packed layout/count as their sources.
 * read_down_ready_trace additionally waits for outgoing returns when configured;
 * the embedder must also coordinate incoming completion across local ranks.
 */
RDMA_PROXY_C_API int rdma_proxy_configure_return_copies(
    RdmaProxyHandle* handle, const RdmaProxyReturnCopy* copies, uint64_t count,
    const int64_t* host_valid_rows, uint64_t row_count, uint64_t width,
    uint64_t tile_n, uint64_t group_tiles, uint64_t element_bytes,
    int32_t per_group, uintptr_t down_done_event);
/* Call after recording this iteration's down_done_event, never before. */
RDMA_PROXY_C_API int rdma_proxy_arm_return_copies(RdmaProxyHandle* handle, uint64_t iteration);

/* CPU DRAM readiness snapshot, enabled by nvlink_return_completion_notifications_enabled.
 * prefix=N means groups [0,N) arrived from every other local GPU. No final
 * reduction or reverse RDMA is implied. Requires the matching prepared iteration.
 * With the feature enabled, read_down_ready_trace also waits for incoming
 * notifications to be consumed and outgoing notifications to be dispatched
 * and consumed by their destinations.
 */
RDMA_PROXY_C_API int rdma_proxy_get_return_ready_prefix(
    RdmaProxyHandle* handle, uint64_t iteration, int32_t* prefix);

/* Available after configure_down_ready_trace with
 * nvlink_return_ready_gpu_publication_enabled=true. Borrowed int32 flags:
 * host_address is pinned CPU memory, device_address is separate GPU HBM.
 * Host reads must be atomic/acquire; GPU consumers must use a suitable
 * acquire polling protocol. Neither address may be modified by the embedder.
 * The host value denotes enqueued publication; the device update is async.
 * prepare_iteration resets both flags to zero and completes the GPU reset
 * before returning; read_down_ready_trace waits for final GPU publication.
 * Finish all GPU consumers before the next preparation or shutdown. Both
 * addresses remain stable until shutdown, when their allocations are freed.
 */
RDMA_PROXY_C_API int rdma_proxy_get_return_ready_publication(
    RdmaProxyHandle* handle, uintptr_t* host_address, uintptr_t* device_address);

RDMA_PROXY_C_API int rdma_proxy_shutdown(RdmaProxyHandle* handle);
RDMA_PROXY_C_API void rdma_proxy_destroy(RdmaProxyHandle* handle);

/* Convenience entry point equivalent to create/initialize/run/shutdown/destroy. */
RDMA_PROXY_C_API int rdma_proxy_run_argv(
    int argc,
    const char* const* argv);

/*
 * Returns the most recent error for the calling thread. The pointer remains
 * valid until the next C API call on that thread.
 */
RDMA_PROXY_C_API const char* rdma_proxy_last_error(void);

#ifdef __cplusplus
}
#endif
