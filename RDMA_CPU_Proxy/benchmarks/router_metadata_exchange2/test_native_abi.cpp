#include "native_proxy.h"
#include <cstddef>
#include <iostream>

// Compare the actual public C layout with ctypes without a CUDA runtime.
int main() {
    std::cout << "size " << sizeof(Rme2Stats) << '\n';
#define FIELD(name) std::cout << #name << ' ' << offsetof(Rme2Stats, name) << '\n'
    FIELD(rdma_work_complete_ms);
    FIELD(local_staging_gpu_ms);
    FIELD(local_staging_wait_ms);
    FIELD(received_index_bytes);
    FIELD(x3_x4_received_bytes);
#undef FIELD
}
