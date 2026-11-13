#include "router_routing.hpp"
#include "cuda_buffers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rdma_proxy;

static void require(bool condition) {
    if (!condition) throw std::runtime_error("route-weight check failed");
}

template <class F> static void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected);
}

static void check_source(ProxyConfig config) {
    RouterRouting routing(config), repeated(config);
    routing.initialize();
    repeated.initialize();
    const auto& experts = routing.original_experts();
    const auto& weights = routing.original_weights();
    require(experts == repeated.original_experts());
    require(weights == repeated.original_weights());
    require(experts.size() == config.num_tokens * config.router_top_k);
    for (std::size_t token = 0; token < config.num_tokens; ++token) {
        double sum = 0;
        for (int k = 0; k < config.router_top_k; ++k) {
            const auto value = weights[token * config.router_top_k + k];
            require(std::isfinite(value) && value > 0 && value <= 1);
            sum += value;
        }
        require(std::abs(sum - 1.0) < 1e-6);
    }

    // Construct destination rows independently from original assignment sets.
    // This deliberately does not use the native weight-to-A_idx alignment helper.
    std::vector<int> seen(experts.size(), 0);
    for (int node = 0; node < config.num_nodes; ++node) {
        const auto& x3 = routing.token_indices_for_node(node);
        for (int gpu = 0; gpu < config.num_gpus_per_node; ++gpu) {
            const auto& metadata = routing.expert_metadata_for_gpu(node, gpu);
            const auto wire = serialize_router_expert_metadata(metadata);
            const auto decoded = deserialize_router_expert_metadata(wire, config.num_tokens, experts.size());
            require(decoded.weighted_contributions && decoded.top_k == config.router_top_k);
            require(decoded.expert_route_weights == metadata.expert_route_weights);
            require(decoded.expert_route_weights.size() == metadata.expert_token_indices.size());
            std::vector<std::size_t> input_to_original;
            for (auto original : x3) {
                bool selected = gpu == config.local_gpu_index;
                for (int k = 0; k < config.router_top_k; ++k)
                    selected |= experts[original * config.router_top_k + k] / routing.experts_per_gpu() ==
                        node * config.num_gpus_per_node + gpu;
                if (selected) input_to_original.push_back(original);
            }
            for (int e = 0; e < routing.experts_per_gpu(); ++e) {
                for (int32_t i = metadata.expert_offsets[e]; i < metadata.expert_offsets[e + 1]; ++i) {
                    const auto original = input_to_original.at(metadata.expert_token_indices[i]);
                    const auto begin = experts.begin() + original * config.router_top_k;
                    const auto found = std::find(begin, begin + config.router_top_k, metadata.first_global_expert + e);
                    require(found != begin + config.router_top_k);
                    const auto source_index = static_cast<std::size_t>(found - experts.begin());
                    require(metadata.expert_route_weights[i] == weights[source_index]);
                    ++seen[source_index];
                }
            }
        }
    }
    require(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
    auto other = config;
    other.node_rank = 1 - config.node_rank;
    RouterRouting another_node(other);
    another_node.initialize();
    require(another_node.original_weights() != weights);
    other = config;
    other.local_gpu_index = 1 - config.local_gpu_index;
    RouterRouting another_gpu(other);
    another_gpu.initialize();
    require(another_gpu.original_weights() != weights);
}

static void check_protocol() {
    RouterExpertMetadata m;
    m.source_node_rank = m.destination_node_rank = 0;
    m.source_gpu_index = m.destination_gpu_index = 0;
    m.num_nodes = 1;
    m.num_gpus_per_node = 2;
    m.num_experts = 2;
    m.experts_per_gpu = 1;
    m.first_global_expert = 0;
    m.top_k = 1;
    m.num_tokens = 5;
    m.expert_offsets = {0, 5};
    m.expert_token_indices = {0, 1, 2, 3, 4};
    m.weighted_contributions = true;
    m.expert_route_weights = {0.0f, 1.0f, 0.123456791f,
        std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max()};
    const auto wire = serialize_router_expert_metadata(m);
    auto decoded = deserialize_router_expert_metadata(wire, 5, 5);
    require(std::memcmp(m.expert_route_weights.data(), decoded.expert_route_weights.data(), 5 * sizeof(float)) == 0);
    rejects([&] { deserialize_router_expert_metadata(wire, 5, 4); });
    rejects([&] { deserialize_router_expert_metadata(wire, 4, 5); });
    rejects([&] { deserialize_router_expert_metadata(wire + " extra", 5, 5); });
    rejects([&] { deserialize_router_expert_metadata(wire.substr(0, wire.rfind(' ')), 5, 5); });
    auto wrong_version = wire;
    wrong_version.replace(wrong_version.find("v2"), 2, "v1");
    rejects([&] { deserialize_router_expert_metadata(wrong_version, 5, 5); });
    const auto last = wire.rfind(' ');
    for (const char* bad_bits : {"2143289344", "2139095040", "3212836864", "4294967296"})
        rejects([&] { deserialize_router_expert_metadata(wire.substr(0, last + 1) + bad_bits, 5, 5); });
    for (float bad : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        auto invalid = m;
        invalid.expert_route_weights[0] = bad;
        rejects([&] { serialize_router_expert_metadata(invalid); });
    }
    auto invalid = m;
    invalid.expert_route_weights.pop_back();
    rejects([&] { serialize_router_expert_metadata(invalid); });
    invalid = m;
    invalid.weighted_contributions = false;
    rejects([&] { serialize_router_expert_metadata(invalid); });
    // Reject configuration disagreement before accepting source metadata.
    ProxyConfig config;
    config.mock_mode = true;
    config.router_routing_enabled = config.nvlink_forwarding_enabled = true;
    config.num_nodes = 1;
    config.num_gpus_per_node = 2;
    config.router_num_experts = 2;
    config.router_top_k = 1;
    config.num_tokens = 5;
    config.token_dimension = 8;
    CudaBuffers buffers(config);
    buffers.initialize();
    rejects([&] { buffers.install_expert_metadata(m); });
    m.expert_route_weights.clear();
    m.weighted_contributions = false;
    m.top_k = 2;
    rejects([&] { buffers.install_expert_metadata(m); });
}

int main(int argc, char** argv) {
    const bool cuda = argc == 2 && std::string(argv[1]) == "--cuda";
    if (cuda) {
#if RDMA_PROXY_HAVE_CUDA
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
#else
        return 77;
#endif
    }
    ProxyConfig config;
    config.mock_mode = !cuda;
    config.router_routing_enabled = config.router_weighted_contributions_enabled = true;
    config.num_nodes = 2;
    config.num_gpus_per_node = 2;
    config.router_num_experts = 16;
    config.router_top_k = 5;
    config.num_tokens = 257;
    for (int node = 0; node < 2; ++node) for (int gpu = 0; gpu < 2; ++gpu) {
        config.node_rank = node;
        config.local_gpu_index = gpu;
        check_source(config);
    }
    check_protocol();
    std::cout << "test_route_weights passed\n";
}
