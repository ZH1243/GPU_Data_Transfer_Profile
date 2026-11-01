#include "cuda_buffers.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rdma_proxy;
static void require(bool value) {
    if (!value) throw std::runtime_error("CPU scatter check failed");
}

static void run(bool per_entry, uint64_t tile, bool cuda) {
    ProxyConfig config;
    config.mock_mode = !cuda;
    config.router_routing_enabled = true;
    config.nvlink_forwarding_enabled = true;
    config.nvlink_forward_completion_notifications_enabled = true;
    config.nvlink_forward_notification_flush_per_entry_enabled = per_entry;
    config.num_nodes = 1;
    config.num_gpus_per_node = 3;
    config.router_num_experts = 6;
    config.router_top_k = 2;
    config.num_tokens = 10;
    config.token_dimension = 256;
    config.dtype = DataType::kBF16;
    config.expert_gemm_m_tile = 2;
    config.expert_gemm_cluster_m = 1;
    config.expert_gemm_n_tile = 128;
    config.expert_gemm_dimension = 512;
    config.expert_gemm_max_swizzle_size = 2; // repeated N groups
    CudaBuffers buffers(config);
    buffers.initialize();
    const std::vector<std::vector<int32_t>> indices{{2,5,9,5,9}, {1,7,1}, {}};
    const std::vector<std::vector<int32_t>> offsets{{0,3,5}, {0,2,3}, {0,0,0}};
    for (int source = 0; source < 3; ++source) {
        RouterExpertMetadata m;
        m.destination_node_rank = 0;
        m.destination_gpu_index = 0;
        m.source_node_rank = 0;
        m.source_gpu_index = source;
        m.num_nodes = 1;
        m.num_gpus_per_node = 3;
        m.num_experts = 6;
        m.experts_per_gpu = 2;
        m.first_global_expert = 0;
        m.num_tokens = 10;
        m.expert_token_indices = indices[source];
        m.expert_offsets = offsets[source];
        buffers.install_expert_metadata(m);
    }
    const std::size_t columns = tile ? 2 : 1;
    std::vector<int64_t> scatter(8*columns, -1), counts(3), inverse(30);
    void* device_scatter = scatter.data();
#if RDMA_PROXY_HAVE_CUDA
    if (cuda) require(cudaMalloc(&device_scatter, scatter.size()*sizeof(int64_t)) == cudaSuccess);
#endif
    int32_t pairs[]{0,0,0,1,0,2};
    uint64_t bases[]{0x10000, 0x20000, 0x30000};
    RdmaProxyCpuScatterConfig setup{sizeof(setup), 3, 8, 10, 256, tile, 2,
        reinterpret_cast<uintptr_t>(device_scatter), pairs, bases, counts.data(), inverse.data()};
    auto bad = setup;
    bad.routes++;
    bool rejected = false;
    try { buffers.configure_cpu_scatter(bad); } catch (const std::exception&) { rejected = true; }
    require(rejected);
    buffers.configure_cpu_scatter(setup);
    require(counts == std::vector<int64_t>({3,2,0}));
    require(inverse == std::vector<int64_t>({2,5,9,-1,-1,-1,-1,-1,-1,-1,
                                          1,7,-1,-1,-1,-1,-1,-1,-1,-1,
                                          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1}));
    const auto& publication = buffers.router_notification_publication_buffers();
    std::vector<int64_t> first_iteration;
    for (uint64_t iteration = 0; iteration < 2; ++iteration) {
        buffers.begin_router_notification_iteration(iteration);
        std::fill(scatter.begin(), scatter.end(), -1);
#if RDMA_PROXY_HAVE_CUDA
        if (cuda) {
            require(cudaMemset(device_scatter, 0xff, scatter.size()*sizeof(int64_t)) == cudaSuccess);
            require(cudaDeviceSynchronize() == cudaSuccess);
        }
#endif
        const auto check_published = [&]() {
            int32_t ready = 0;
            auto* table = static_cast<const int32_t*>(publication.device_table.ptr);
            std::vector<int32_t> table_copy;
            if (!cuda) ready = *static_cast<const int32_t*>(publication.device_ready_rows.ptr);
#if RDMA_PROXY_HAVE_CUDA
            else {
                buffers.synchronize_router_notification_publication();
                table_copy.resize(publication.device_table.bytes / sizeof(int32_t));
                require(cudaMemcpy(&ready, publication.device_ready_rows.ptr, sizeof(ready),
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                require(cudaMemcpy(table_copy.data(), table, publication.device_table.bytes,
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                require(cudaMemcpy(scatter.data(), device_scatter, scatter.size()*sizeof(int64_t),
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                table = table_copy.data();
            }
#endif
            require(ready == static_cast<int32_t>(
                buffers.router_computation_scheduler_state().published_rows));
            std::vector<int> seen(8);
            for (int r = 0; r < ready; r += publication.n_groups_per_m_cluster) {
                auto* row = table + r * publication.table_width;
                std::size_t packed = 0;
                for (int s = 0; s < 3; ++s) packed += row[2+2*s];
                for (int s = 0; s < 3; ++s) {
                    for (int i = row[2+2*s]; i < row[3+2*s]; ++i, ++packed) {
                        require(packed < 8 && !seen[packed]++);
                        const auto input = indices[s][i];
                        const auto begin = inverse.begin() + s*10;
                        const auto compact = std::find(begin, begin + counts[s], input) - begin;
                        require(compact < counts[s]);
                        require(scatter[packed*columns] == static_cast<int64_t>(
                            bases[s] + compact*(tile ? tile : 256)*2));
                        if (tile) require(scatter[packed*columns+1] == counts[s]*tile*2);
                    }
                }
            }
            if (static_cast<std::size_t>(ready) == publication.table_rows)
                require(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
        };
        // Reversing source arrival changes packed positions, but destinations
        // and duplicate-route reduction addresses must remain correct.
        for (int s : (iteration == 0 ? std::vector<int>{0,1,2} : std::vector<int>{1,0,2})) {
            buffers.process_router_notification_completion(0, s, iteration, 0, 10);
            check_published();
        }
        require(buffers.router_computation_scheduler_state().published_rows == publication.table_rows);
        if (!iteration) first_iteration = scatter;
        else require(scatter != first_iteration);
    }
#if RDMA_PROXY_HAVE_CUDA
    if (cuda) require(cudaFree(device_scatter) == cudaSuccess);
#endif
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
    for (bool per_entry : {false, true}) for (uint64_t tile : {0, 128}) run(per_entry, tile, cuda);
    std::cout << "test_cpu_scatter passed\n";
}
