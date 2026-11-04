#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI 5; Linux CUDA only. All functions except destroy/error return 0 on success.
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
    uint64_t received_index_bytes;
    uint64_t peer_received_index_bytes;
    uint64_t header_bytes;
} Rme2Stats;
int rme2_abi_version(void);
const char* rme2_error(void);
int rme2_create(int rank, int device, int tokens, const char* shm_name,
                double timeout_seconds, int validate_indices, int cpu_core,
                int per_source_readback, int metadata_barrier, Rme2Proxy** out);
int rme2_export(Rme2Proxy*, void* handle64);
int rme2_connect(Rme2Proxy*, const void* handles_8_by_64);
int rme2_unlink(Rme2Proxy*);
/* Collectively disconnect, barrier, then destroy: imported mappings must close
 * before any owner frees its IPC arena. */
int rme2_disconnect(Rme2Proxy*);
/* Setup-only capture on a nondefault stream. The proxy owns the executable;
 * caller owns all captured tensors and must keep them and submission streams
 * alive through final wait/disconnect/destroy. Capture must contain only metadata.
 * After capture, submit schedules metadata AND exchange on the native worker.
 */
int rme2_capture_begin(Rme2Proxy*, void* cuda_stream);
int rme2_capture_end(Rme2Proxy*, int launch_barrier);
int rme2_submit(Rme2Proxy*, uint64_t iteration, const int32_t* offsets,
                const int32_t* indices, void* cuda_stream);
int rme2_wait(Rme2Proxy*, Rme2Stats*);
int rme2_view(Rme2Proxy*, int source, const int32_t** offsets,
              const int32_t** indices, int* count);
void rme2_destroy(Rme2Proxy*);
#ifdef __cplusplus
}
#endif
