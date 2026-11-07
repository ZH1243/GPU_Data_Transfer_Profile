#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI 11; Linux CUDA only. All functions except destroy/error return 0 on success.
 * create/connect/submit/wait/view must be called serially by one host thread.
 * submit queues work; the owned native thread handles readback and installation.
 * Pointers returned by view are borrowed until the next submit or destroy.
 */
typedef struct Rme2Proxy Rme2Proxy;
typedef struct {
    double metadata_launch_barrier_ms;
    double metadata_gpu_ms;
    double metadata_completion_wait_ms;
    double metadata_barrier_wait_ms;
    double exchange_gpu_ms;
    double sender_completion_wait_ms;
    double peer_completion_wait_ms;
    double headers_d2h_ms;
    double payload_d2h_ms;
    double payload_checks_ms;
    double payload_submit_ms;
    double payload_event_record_ms;
    double payload_completion_ms;
    double payload_other_ms;
    double install_ms;
    double native_ms;
    double rdma_post_ms;
    double rdma_poll_ms;
    double rdma_post_recv_ms;
    double rdma_work_complete_ms;
    double local_staging_gpu_ms;
    double local_staging_wait_ms;
    double staging_before_d2h_wait_ms;
    uint64_t received_index_bytes;
    uint64_t peer_received_index_bytes;
    uint64_t header_bytes;
    uint64_t rdma_sent_bytes;
    uint64_t rdma_received_bytes;
    /* x3/x4 payload only (5 bytes/token); RDMA totals above include its header. */
    uint64_t x3_x4_sent_bytes;
    uint64_t x3_x4_received_bytes;
} Rme2Stats;
int rme2_abi_version(void);
int rme2_set_batch_payload_copy(Rme2Proxy*, int enabled);
/* Setup-only: gate all local header/payload D2H on local staging completion.
 * Requires a staging graph before submit; exchange publication is unchanged. */
int rme2_set_d2h_after_local_staging(Rme2Proxy*, int enabled);
/* Setup-only: defer gather launch until all local payload D2H completes.
 * Mutually exclusive with d2h_after_local_staging; requires a staging graph. */
int rme2_set_local_staging_after_d2h(Rme2Proxy*, int enabled);
const char* rme2_error(void);
int rme2_create(int rank, int device, int tokens, const char* shm_name,
                double timeout_seconds, int validate_indices, int cpu_core,
                int per_source_readback, int metadata_barrier, int nodes, int node, Rme2Proxy** out);
/* Two-node setup only, before connect/capture/submit. Descriptor is NUL-terminated.
 * init registers relay-owned HBM and final pinned host records once. */
int rme2_rdma_init(Rme2Proxy*, const char* device, int port, int gid,
                   char* descriptor, int descriptor_capacity);
int rme2_rdma_connect(Rme2Proxy*, const char* descriptor);
int rme2_export(Rme2Proxy*, void* handle64);
int rme2_connect(Rme2Proxy*, const void* handles_8_by_64);
int rme2_unlink(Rme2Proxy*);
/* Collectively disconnect, barrier, then destroy: imported mappings must close
 * before any owner frees its IPC arena. */
int rme2_disconnect(Rme2Proxy*);
/* Setup-only capture on a nondefault stream. The proxy owns the executable;
 * caller owns all captured tensors and must keep them and submission streams
 * alive through final wait/disconnect/destroy. Capture must contain only metadata.
 * After capture, submit schedules metadata, exchange and optional staging on
 * the native worker.
 */
int rme2_capture_begin(Rme2Proxy*, void* cuda_stream);
int rme2_capture_end(Rme2Proxy*, int launch_barrier);
/* Optional setup-only capture of the local token-row gather, separate from
 * metadata capture. Works with Python eager/graph and native metadata launch.
 * By default submit enqueues this graph AFTER the exchange completion event on the same
 * stream. By default CPU readback/RDMA waits only for exchange; the optional
 * d2h_after_local_staging mode gates readback on staging too. Conversely,
 * local_staging_after_d2h defers graph launch until all local D2H completes.
 * wait drains staging in every mode.
 * Caller keeps captured tensors alive through disconnect/destroy. */
int rme2_staging_capture_begin(Rme2Proxy*, void* cuda_stream);
int rme2_staging_capture_end(Rme2Proxy*);
/* Two-node submissions require node_offsets/x3/x4; single-node ignores them.
 * All input buffers must remain alive and unchanged until wait completes. */
int rme2_submit(Rme2Proxy*, uint64_t iteration, const int32_t* offsets,
                const int32_t* indices, const int32_t* node_offsets,
                const int32_t* x3, const uint8_t* x4, void* cuda_stream);
int rme2_wait(Rme2Proxy*, Rme2Stats*);
int rme2_view(Rme2Proxy*, int source, const int32_t** offsets,
              const int32_t** indices, int* count);
/* Two-node only: remote same-index source's slice for this node, borrowed
 * through the next submit/destroy, available only after successful wait. */
int rme2_x3_x4_view(Rme2Proxy*, const int32_t** x3, const uint8_t** x4, int* count);
void rme2_destroy(Rme2Proxy*);
#ifdef __cplusplus
}
#endif
