#pragma once

#include "config.hpp"
#include "protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rdma_proxy {

class RouterRouting {
public:
    explicit RouterRouting(ProxyConfig config);
    ~RouterRouting();

    RouterRouting(const RouterRouting&) = delete;
    RouterRouting& operator=(const RouterRouting&) = delete;

    void initialize();
    const std::vector<std::size_t>& token_indices_for_node(int node_rank) const;
    const std::vector<uint8_t>& token_masks_for_node(int node_rank) const;
    const RouterExpertMetadata& expert_metadata_for_gpu(
        int node_rank,
        int local_gpu_index) const;
    int experts_per_gpu() const;

private:
    void initialize_mock();
    void initialize_cuda();
    void release_cuda() noexcept;

    ProxyConfig config_;
    std::vector<std::vector<std::size_t>> token_indices_by_node_;
    std::vector<std::vector<uint8_t>> token_masks_by_node_;
    std::vector<RouterExpertMetadata> expert_metadata_by_gpu_;
    std::vector<void*> device_allocations_;
    int32_t* pinned_x3_{nullptr};
    uint8_t* pinned_x4_{nullptr};
    int32_t* pinned_node_offsets_{nullptr};
    bool initialized_{false};
};

}  // namespace rdma_proxy
