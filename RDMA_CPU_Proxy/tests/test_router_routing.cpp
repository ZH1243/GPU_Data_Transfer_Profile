#include "router_routing.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <vector>

int main() {
    rdma_proxy::ProxyConfig config;
    config.mock_mode = true;
    config.router_routing_enabled = true;
    config.num_nodes = 2;
    config.num_gpus_per_node = 2;
    config.local_gpu_index = 0;
    config.num_tokens = 64;
    config.router_num_experts = 16;
    config.router_top_k = 4;
    config.router_seed = 1234;

    rdma_proxy::RouterRouting routing(config);
    routing.initialize();
    assert(routing.experts_per_gpu() == 4);

    std::size_t total_expert_routes = 0;
    for (int node = 0; node < config.num_nodes; ++node) {
        const auto& masks = routing.token_masks_for_node(node);
        for (int gpu = 0; gpu < config.num_gpus_per_node; ++gpu) {
            const auto& metadata = routing.expert_metadata_for_gpu(node, gpu);
            assert(metadata.source_node_rank == config.node_rank);
            assert(metadata.source_gpu_index == config.local_gpu_index);
            assert(metadata.destination_node_rank == node);
            assert(metadata.destination_gpu_index == gpu);
            assert(metadata.experts_per_gpu == routing.experts_per_gpu());
            assert(metadata.expert_offsets.size() ==
                   static_cast<std::size_t>(routing.experts_per_gpu() + 1));
            assert(metadata.expert_offsets.front() == 0);
            assert(static_cast<std::size_t>(metadata.expert_offsets.back()) ==
                   metadata.expert_token_indices.size());
            assert(std::is_sorted(
                metadata.expert_offsets.begin(), metadata.expert_offsets.end()));

            int bit = config.local_gpu_index - gpu;
            if (bit < 0) bit += config.num_gpus_per_node;
            const auto gpu_mask = static_cast<uint8_t>(1U << bit);
            const auto destination_token_count = static_cast<int32_t>(std::count_if(
                masks.begin(), masks.end(),
                [&](uint8_t mask) { return (mask & gpu_mask) != 0; }));
            for (const auto index : metadata.expert_token_indices) {
                assert(index >= 0);
                if (gpu == config.local_gpu_index) {
                    assert(static_cast<std::size_t>(index) < masks.size());
                    assert((masks[static_cast<std::size_t>(index)] & gpu_mask) != 0);
                } else {
                    assert(index < destination_token_count);
                }
            }
            for (int expert = 0; expert < routing.experts_per_gpu(); ++expert) {
                const auto begin = metadata.expert_offsets[static_cast<std::size_t>(expert)];
                const auto end = metadata.expert_offsets[static_cast<std::size_t>(expert + 1)];
                assert(std::is_sorted(
                    metadata.expert_token_indices.begin() + begin,
                    metadata.expert_token_indices.begin() + end));
            }
            total_expert_routes += metadata.expert_token_indices.size();
        }
    }
    assert(total_expert_routes ==
           config.num_tokens * static_cast<std::size_t>(config.router_top_k));

    rdma_proxy::RouterRouting same_routing(config);
    same_routing.initialize();
    for (int node = 0; node < config.num_nodes; ++node) {
        const auto& indices = routing.token_indices_for_node(node);
        const auto& masks = routing.token_masks_for_node(node);
        assert(indices == same_routing.token_indices_for_node(node));
        assert(masks == same_routing.token_masks_for_node(node));
        assert(masks.size() == indices.size());
        assert(std::is_sorted(masks.begin(), masks.end(), std::greater<uint8_t>()));
        assert(indices.size() <= config.num_tokens);
        auto sorted = indices;
        std::sort(sorted.begin(), sorted.end());
        assert(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
        for (const auto token : indices) assert(token < config.num_tokens);
        for (const auto mask : masks) assert(mask != 0 && (mask & ~0x3U) == 0);
    }

    auto different_config = config;
    different_config.router_seed = 9876;
    rdma_proxy::RouterRouting different_routing(different_config);
    different_routing.initialize();
    bool any_node_differs = false;
    for (int node = 0; node < config.num_nodes; ++node) {
        any_node_differs = any_node_differs ||
            routing.token_indices_for_node(node) != different_routing.token_indices_for_node(node);
    }
    assert(any_node_differs);

    std::cout << "test_router_routing passed\n";
    return 0;
}
