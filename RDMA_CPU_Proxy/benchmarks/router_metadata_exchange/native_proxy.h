#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/* ABI v2. Strings and this struct must stay alive throughout run().
 * One synchronous run per process; the caller may invoke it on a worker thread.
 * All exceptions are translated into a nonzero status and a bounded error string.
 */
typedef struct RouterMetadataBenchConfig {
    int32_t rank, world, gpus_per_node, port_base, node_port_stride;
    int32_t num_tokens, num_experts, top_k, token_dimension, element_bytes;
    int32_t tokens_per_chunk, datasets, warmup, iterations, preserialized, verify, binary;
    uint64_t seed;
    double timeout;
    const char* hosts;
    const char* bind_host;
    const char* output;
} RouterMetadataBenchConfig;

uint32_t router_metadata_bench_abi_version(void);
int router_metadata_bench_run(const RouterMetadataBenchConfig* config,
                             char* error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
