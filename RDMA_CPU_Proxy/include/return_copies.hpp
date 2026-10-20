#pragma once

#include "cuda_buffers.hpp"
#include "rdma_proxy_c_api.h"
#include <map>
#include <vector>

namespace rdma_proxy {

// Called only under DownReadyTrace's mutex. The receiver submits/polls; the
// embedder arms only after recording this iteration's down-completion event.
class ReturnCopies {
public:
    ReturnCopies(int device, bool mock, int groups,
                 const RdmaProxyReturnCopy* copies, std::size_t count,
                 const int64_t* rows, std::size_t row_count,
                 std::size_t width, std::size_t tile, std::size_t group_tiles,
                 std::size_t element_bytes, bool per_group, uintptr_t down_done);
    ~ReturnCopies();
    void reset();
    void arm();
    bool poll(int ready);
    void drain();

private:
    void submit(int group);
    int device_, groups_, submitted_{0};
    bool mock_, per_group_, armed_{false};
    const int64_t* rows_;
    std::size_t width_, tile_, group_tiles_, element_bytes_;
    uintptr_t down_done_;
    void* stream_{nullptr};
    std::map<int, std::vector<RdmaProxyReturnCopy>> peers_;
    std::map<int, CudaPreparedForwardBatch> batches_;
};
}  // namespace rdma_proxy
