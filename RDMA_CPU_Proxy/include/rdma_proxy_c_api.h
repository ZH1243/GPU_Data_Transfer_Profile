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

/* Optional CPU scatter setup, after initialization and before iteration zero.
 * source_pairs is host int32[num_sources,2] in gather input order; output_bases
 * is host uint64[num_sources]. Host outputs are counts[num_sources] and
 * output_to_input[num_sources,capacity] (-1 in unused tails). All host arrays
 * are used synchronously. scatter_addresses is borrowed CUDA int64[routes,2]
 * for tile_n > 0, or int64[routes] for row-major; retain it and output buffers
 * through shutdown. Routing and output allocations must remain fixed.
 */
typedef struct RdmaProxyCpuScatterConfig {
    uint64_t struct_size;
    uint64_t num_sources, routes, capacity, width, tile_n, element_bytes;
    uintptr_t scatter_addresses;
    const int32_t* source_pairs;
    const uint64_t* output_bases;
    int64_t* counts;
    int64_t* output_to_input;
} RdmaProxyCpuScatterConfig;

/*
 * Stable C ABI for embedding the C++ proxy in another process (for example,
 * a Python worker using ctypes). The handle is intentionally opaque so C++
 * types and standard-library objects never cross the shared-library boundary.
 */
typedef struct RdmaProxyHandle RdmaProxyHandle;
RDMA_PROXY_C_API int rdma_proxy_configure_cpu_scatter(
    RdmaProxyHandle* handle, const RdmaProxyCpuScatterConfig* config);

/* Optional weighted CPU scatter setup: call after configure_cpu_scatter and
 * before iteration zero. Borrowed FP32[routes] storage must survive shutdown.
 * Each selected range is copied once before its gather readiness publication,
 * unless deferred publication is configured below. */
RDMA_PROXY_C_API int rdma_proxy_configure_cpu_route_weights(
    RdmaProxyHandle* handle, uintptr_t packed_weights, uint64_t routes);

/* Opt-in one-copy CPU metadata publication. Call after scatter and optional
 * weights setup, before iteration zero. scatter_addresses begins a 16-byte
 * aligned borrowed allocation. Its exact size is routes*columns*8 unweighted;
 * weighted storage places FP32[routes] at round_up(scatter_bytes,16), and bytes
 * includes that weight array. Retain the entire allocation through shutdown.
 * After final gather-ready publication is enqueued, CPU staging is reconstructed
 * from the completed host gather table, once per M batch. One slab H2D copy and
 * event record then follow on the publication stream.
 * The existing per-batch behavior remains default unless this API is called. */
RDMA_PROXY_C_API int rdma_proxy_configure_cpu_scatter_deferred(
    RdmaProxyHandle* handle, uint64_t bytes);
/* Alternative opt-in policy with the same contiguous slab layout. One persistent
 * native CPU worker consumes committed host gather prefixes while notification
 * publication continues. It packs each complete M-batch descriptor once, outside
 * notification locks; one slab H2D/event follows final gather submission and all
 * packing. Configure exactly one of deferred/threaded, before iteration zero.
 * Native shutdown joins the worker and retires its DMA before returning. */
RDMA_PROXY_C_API int rdma_proxy_configure_cpu_scatter_threaded(
    RdmaProxyHandle* handle, uint64_t bytes);
/* Host-wait, bounded by timeout_ms > 0, until this prepared iteration's event
 * RECORD is submitted, then enqueue cudaStreamWaitEvent on a nonzero caller
 * stream. Supports deferred and threaded modes; worker errors are reported.
 * This does not synchronize the GPU. Call before down GEMM submission;
 * another host thread must continue driving notification/communication work.
 * Join compute consumers before preparing the next iteration or shutting down. */
RDMA_PROXY_C_API int rdma_proxy_wait_cpu_scatter_ready(
    RdmaProxyHandle* handle, uint64_t iteration, uintptr_t cuda_stream, uint64_t timeout_ms);

/* Original source [num_tokens,top_k] assignments, before destination row mapping.
 * Query with both arrays NULL and capacity=0; otherwise copy both host arrays.
 * Weights are normalized FP32 when enabled, exactly one when disabled.
 * These immutable arrays allow references independent of gather/scatter packing. */
RDMA_PROXY_C_API int rdma_proxy_get_local_router_assignments(
    RdmaProxyHandle* handle, int32_t* experts, float* weights, uint64_t capacity,
    uint64_t* count, uint64_t* num_tokens, uint64_t* top_k);

/* Device-buffer kinds requested through RdmaProxyDeviceBufferAllocator. */
typedef enum RdmaProxyDeviceBufferKind {
    RDMA_PROXY_DEVICE_BUFFER_RDMA_SEND = 1,
    RDMA_PROXY_DEVICE_BUFFER_RDMA_RECEIVE = 2,
    RDMA_PROXY_DEVICE_BUFFER_NVLINK_RECEIVE = 3,
    RDMA_PROXY_DEVICE_BUFFER_ROUTER_A_IDX = 4,
    RDMA_PROXY_DEVICE_BUFFER_GATHER_TABLE = 5,
    RDMA_PROXY_DEVICE_BUFFER_GATHER_READY_ROWS = 6,
    RDMA_PROXY_DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS = 7,
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
/* Static routing masks in this GPU's full staging (local source_node) or RDMA
 * receive (remote source_node) row order. Bit g means computing GPU g uses row.
 * Query length with masks=NULL, capacity=0; populated rows exclude capacity tails.
 * Valid after initialization; benchmark routing remains fixed across iterations. */
RDMA_PROXY_C_API int rdma_proxy_get_intra_node_source_masks(
    RdmaProxyHandle* handle, int32_t source_node, uint8_t* masks,
    uint64_t capacity, uint64_t* count);
RDMA_PROXY_C_API int rdma_proxy_finish(RdmaProxyHandle* handle);

/* This source GPU's original token IDs in outgoing x3 order for a destination
 * node, including the local staging domain. Query with indices=NULL, capacity=0.
 * num_tokens is the full original input size, including tokens absent from x3.
 * Routing is immutable after initialization in this benchmark. */
RDMA_PROXY_C_API int rdma_proxy_get_outgoing_token_indices(
    RdmaProxyHandle* handle, int32_t destination_node, uint64_t* indices,
    uint64_t capacity, uint64_t* count, uint64_t* num_tokens);

/* Reverse RDMA: caller owns the stable tiled source/destination GPU storage.
 * Empty buffers still supply a non-null one-element allocation for registration.
 * Setup is collective among same-index GPUs across nodes, before iterations.
 * Source rows are incoming remote x3 rows; destination rows are outgoing x3 rows.
 * Layout is [width/tile_n, rows, tile_n], packed independently per peer.
 * The returned device address aliases a native-owned mapped pinned int32 flag.
 * After all CTA stores for groups [0,N), GPU must system-release store N there.
 * prepare_iteration resets it; run_iteration waits for all outgoing and incoming
 * return group markers, including GPUDirect receive visibility, before returning.
 * Keep GPU storage alive until shutdown. Join all GPU writers before shutdown.
 * Existing QP workers send returns after their current forward task finishes.
 */
typedef struct RdmaProxyNetworkReturnBuffer {
    int32_t peer;
    uintptr_t source;
    uintptr_t destination;
    uint64_t source_rows;
    uint64_t destination_rows;
} RdmaProxyNetworkReturnBuffer;
RDMA_PROXY_C_API int rdma_proxy_get_rdma_return_receive_rows(
    RdmaProxyHandle* handle, int32_t peer, uint64_t* rows);
RDMA_PROXY_C_API int rdma_proxy_configure_rdma_returns(
    RdmaProxyHandle* handle, const RdmaProxyNetworkReturnBuffer* buffers, uint64_t count,
    uint64_t width, uint64_t tile_n, uint64_t group_tiles, uint64_t element_bytes,
    uint64_t chunk_bytes, uint64_t timeout_ms, uintptr_t* mapped_device_prefix);
/* Optional post-iteration SHA256 transport check; same-index peers call in rank order. */
RDMA_PROXY_C_API int rdma_proxy_check_rdma_return(
    RdmaProxyHandle* handle, int32_t peer, const char* sent_sha256, const char* received_sha256);

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
 * prefix=N means groups [0,N) arrived from every other local GPU and this
 * GPU's retained outputs are ready for those groups. No final
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
