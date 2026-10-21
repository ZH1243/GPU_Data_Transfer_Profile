#include "return_copies.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace rdma_proxy {
namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::runtime_error("return-copy size overflow");
    return a * b;
}
}

ReturnCopies::ReturnCopies(
    int device, bool mock, int groups, const RdmaProxyReturnCopy* copies,
    std::size_t count, const int64_t* rows, std::size_t row_count,
    std::size_t width, std::size_t tile, std::size_t group_tiles,
    std::size_t element_bytes, bool per_group, uintptr_t down_done,
    std::function<void(int)> group_completed)
    : device_(device), groups_(groups), mock_(mock), per_group_(per_group),
      rows_(rows), width_(width), tile_(tile),
      group_tiles_(group_tiles), element_bytes_(element_bytes), down_done_(down_done),
      group_completed_(std::move(group_completed)) {
    if (!copies || !count || !rows || !row_count || !width || !element_bytes ||
        groups <= 0 || (!mock && !down_done) || (per_group && (!tile || !group_tiles)))
        throw std::runtime_error("invalid return-copy configuration");
    if (tile) {
        const auto tiles = 1 + (width - 1) / tile;
        if (!group_tiles || 1 + (tiles - 1) / group_tiles != static_cast<std::size_t>(groups))
            throw std::runtime_error("return layout disagrees with ready groups");
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto& copy = copies[i];
        if (!copy.source || !copy.destination || !copy.capacity_bytes ||
            copy.row_index >= row_count || copy.peer < 0)
            throw std::runtime_error("invalid return-copy descriptor");
        peers_[copy.peer].push_back(copy);
    }
    for (const auto& peer : peers_) batches_[peer.first].reserve(peer.second.size());
    stream_ = create_cuda_stream(device, true, mock);
}

ReturnCopies::~ReturnCopies() {
    // Explicit drain during proxy shutdown reports errors; destructor is a fallback.
    try { drain(); } catch (...) {}
    destroy_cuda_stream(stream_, mock_);
}

void ReturnCopies::reset() {
    if (armed_ && (submitted_ != groups_ || !query_cuda_stream(stream_, mock_)))
        throw std::runtime_error("previous return copies are still active");
    submitted_ = 0;
    armed_ = false;
}

void ReturnCopies::arm() {
    if (armed_) throw std::runtime_error("return copies already armed");
    armed_ = true;
}

void ReturnCopies::submit(int group) {
    // Validate/build every peer before submitting any of this group's batches.
    for (const auto& peer : peers_) {
        auto& batch = batches_.at(peer.first);
        batch.clear();
        for (const auto& copy : peer.second) {
            const auto valid = rows_[copy.row_index];
            if (valid < 0) throw std::runtime_error("negative return valid-row count");
            std::size_t bytes = multiply(multiply(static_cast<std::size_t>(valid), width_), element_bytes_);
            std::size_t offset = 0;
            if (tile_) {
                const auto tiles = 1 + (width_ - 1) / tile_;
                const auto tile_bytes = multiply(multiply(static_cast<std::size_t>(valid), tile_), element_bytes_);
                bytes = multiply(tiles, tile_bytes);
                if (bytes > copy.capacity_bytes) throw std::runtime_error("return output exceeds capacity");
                if (group >= 0) {
                    const auto first = multiply(static_cast<std::size_t>(group), group_tiles_);
                    offset = multiply(first, tile_bytes);
                    bytes = multiply(std::min(group_tiles_, tiles - first), tile_bytes);
                }
            }
            if (offset > copy.capacity_bytes || bytes > copy.capacity_bytes - offset)
                throw std::runtime_error("return copy exceeds capacity");
            if (bytes) batch.append(reinterpret_cast<void*>(copy.destination + offset),
                                    reinterpret_cast<const void*>(copy.source + offset), bytes);
        }
    }
    for (auto& batch : batches_)
        launch_cuda_prepared_forward_batch_async(batch.second, stream_, mock_);
}

bool ReturnCopies::poll(int ready) {
    if (!armed_ || ready == 0) return false;
    if (ready < 0 || ready > groups_) throw std::runtime_error("invalid return ready prefix");
    if (per_group_) {
        while (submitted_ < ready) {
            submit(submitted_);
            // Wait for this group's batches to all destination GPUs before
            // submitting another group, even if its ready prefix is visible.
            synchronize_cuda_stream(stream_, mock_);
            if (group_completed_) group_completed_(submitted_);
            ++submitted_;
        }
    } else if (ready == groups_ && submitted_ == 0) {
#if RDMA_PROXY_HAVE_CUDA
        if (!mock_) {
            const auto status = cudaEventQuery(reinterpret_cast<cudaEvent_t>(down_done_));
            if (status == cudaErrorNotReady) return false;
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
#endif
        submit(-1);
        submitted_ = groups_;
    }
    return submitted_ == groups_ && query_cuda_stream(stream_, mock_);
}

void ReturnCopies::drain() {
    select_cuda_device_for_thread(device_, mock_);
    synchronize_cuda_stream(stream_, mock_);
}
}  // namespace rdma_proxy
