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
 * size. Negative peer/source identifiers mean "not applicable".
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
RDMA_PROXY_C_API int rdma_proxy_finish(RdmaProxyHandle* handle);

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
