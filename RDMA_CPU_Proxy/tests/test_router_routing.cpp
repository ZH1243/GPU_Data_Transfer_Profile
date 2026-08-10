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
