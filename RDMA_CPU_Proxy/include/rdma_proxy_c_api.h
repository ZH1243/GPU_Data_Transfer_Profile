#pragma once

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

RDMA_PROXY_C_API int rdma_proxy_initialize(RdmaProxyHandle* handle);
RDMA_PROXY_C_API int rdma_proxy_run(RdmaProxyHandle* handle);
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

