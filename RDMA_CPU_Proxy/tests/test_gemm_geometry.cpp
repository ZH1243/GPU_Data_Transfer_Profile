#include "cuda_buffers.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rdma_proxy;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void check_scheduler(int capacity, std::size_t mma_m, std::size_t cluster_m,
                     std::size_t expected_cta_rows, std::size_t expected_cluster_rows) {
    ProxyConfig config;
    config.mock_mode = true;
    config.router_routing_enabled = true;
    config.nvlink_forwarding_enabled = true;
    config.nvlink_forward_completion_notifications_enabled = true;
    config.num_nodes = 1;
    config.num_gpus_per_node = 3;
    config.router_num_experts = 6;
    config.router_top_k = 2;
    config.num_tokens = expected_cluster_rows + 11;
    config.token_dimension = 256;
    config.expert_gemm_device_capacity = capacity;
    config.expert_gemm_m_tile = mma_m;
    config.expert_gemm_cluster_m = cluster_m;
    config.expert_gemm_n_tile = 128;
    config.expert_gemm_dimension = 768;
    config.expert_gemm_max_swizzle_size = 2;
    require(expert_gemm_cta_m_tile(config) == expected_cta_rows, "CTA row geometry");
    require(expert_gemm_cluster_rows(config) == expected_cluster_rows, "cluster row geometry");

    // Expert zero spans sources and a full plus partial cluster. Expert one is
    // a partial cluster with duplicate tokens across experts; source two is empty.
    const std::array<std::array<std::size_t, 2>, 3> counts{{
        {{expected_cluster_rows / 2 + 3, expected_cluster_rows / 2 + 1}},
        {{expected_cluster_rows + 1, 3}}, {{0, 0}}}};
    std::array<std::vector<int32_t>, 3> indices;
    std::array<std::vector<int32_t>, 3> offsets;
    std::size_t routes = 0;
    CudaBuffers buffers(config);
    buffers.initialize();
    for (int source = 0; source < 3; ++source) {
        offsets[source].push_back(0);
        for (std::size_t expert = 0; expert < 2; ++expert) {
            for (std::size_t token = 0; token < counts[source][expert]; ++token) {
                indices[source].push_back(static_cast<int32_t>(token));
            }
            offsets[source].push_back(static_cast<int32_t>(indices[source].size()));
        }
        routes += indices[source].size();
        RouterExpertMetadata metadata;
        metadata.destination_node_rank = metadata.destination_gpu_index = 0;
        metadata.source_node_rank = 0;
        metadata.source_gpu_index = source;
        metadata.num_nodes = 1;
        metadata.num_gpus_per_node = 3;
        metadata.num_experts = 6;
        metadata.experts_per_gpu = 2;
        metadata.first_global_expert = 0;
        metadata.num_tokens = config.num_tokens;
        metadata.expert_token_indices = indices[source];
        metadata.expert_offsets = offsets[source];
        buffers.install_expert_metadata(metadata);
    }
    const auto& publication = buffers.router_notification_publication_buffers();
    require(publication.table_width == 8, "multi-buffer descriptor width");
    require(publication.n_groups_per_m_cluster == 3, "N group count");
    require(publication.table_rows == 9, "allocation must use per-CTA M geometry");

    // Validate the compact N-tile-major scatter addresses against the source
    // tokens actually consumed by the simulated QuACK CTAs, not only metadata sizes.
    std::vector<int64_t> scatter(routes * 2, -1), compact_counts(3);
    std::vector<int64_t> inverse(3 * config.num_tokens, -1);
    const int32_t pairs[]{0, 0, 0, 1, 0, 2};
    const uint64_t bases[]{0x100000, 0x200000, 0x300000};
    RdmaProxyCpuScatterConfig scatter_config{
        sizeof(scatter_config), 3, routes, config.num_tokens, 256, 128, 2,
        reinterpret_cast<uintptr_t>(scatter.data()), pairs, bases,
        compact_counts.data(), inverse.data()};
    buffers.configure_cpu_scatter(scatter_config);
    for (int source = 0; source < 3; ++source) {
        require(compact_counts[source] == static_cast<int64_t>(counts[source][0]),
                "compact unique-token counts");
    }

    const auto check_publication = [&]() {
        const auto state = buffers.router_computation_scheduler_state();
        const auto ready = *static_cast<const int32_t*>(publication.device_ready_rows.ptr);
        require(ready == static_cast<int32_t>(state.published_rows), "ready prefix");
        const auto* table = static_cast<const int32_t*>(publication.device_table.ptr);
        std::array<std::vector<int>, 3> consumed;
        for (int source = 0; source < 3; ++source) consumed[source].resize(indices[source].size());
        std::vector<int> packed_seen(routes);
        for (int row_index = 0; row_index < ready; row_index += 3) {
            const auto* row = table + row_index * publication.table_width;
            require(row[0] >= 0 && row[0] < 2, "expert identity");
            std::size_t packed_base = 0;
            std::vector<std::pair<int, int>> cluster_routes;
            for (int source = 0; source < 3; ++source) {
                const int start = row[2 + source * 2], end = row[3 + source * 2];
                require(start >= offsets[source][row[0]] &&
                        end <= offsets[source][row[0] + 1] && end >= start,
                        "descriptor expert range");
                packed_base += static_cast<std::size_t>(start);
                for (int index = start; index < end; ++index) {
                    cluster_routes.emplace_back(source, index);
                }
            }
            require(!cluster_routes.empty() && cluster_routes.size() <= expected_cluster_rows,
                    "descriptor exceeds physical cluster row coverage");
            for (int group = 0; group < 3; ++group) {
                const auto* repeated = row + group * publication.table_width;
                require(repeated[0] == row[0] && repeated[1] == group * 2,
                        "N group identity");
                require(std::equal(row + 2, row + 8, repeated + 2), "N group routes");
            }
            // A descriptor larger than the physical cluster span would leave
            // routes unconsumed in the real kernel. Enumerate CTA lanes explicitly.
            for (std::size_t cta = 0; cta < cluster_m; ++cta) {
                for (std::size_t lane = 0; lane < expected_cta_rows; ++lane) {
                    const auto local_row = cta * expected_cta_rows + lane;
                    if (local_row >= cluster_routes.size()) continue;
                    const auto [source, index] = cluster_routes[local_row];
                    require(++consumed[source][index] == 1, "route consumed more than once");
                    const auto packed = packed_base + local_row;
                    require(packed < routes && ++packed_seen[packed] == 1, "packed row collision");
                    const auto token = indices[source][index];
                    require(scatter[packed * 2] == static_cast<int64_t>(bases[source] + token * 128 * 2),
                            "compact scatter row address");
                    require(scatter[packed * 2 + 1] == compact_counts[source] * 128 * 2,
                            "compact scatter N-tile byte stride");
                    // A scalar GEMM reference: gathered source value times the
                    // expert weight. This also checks cross-source packed mapping.
                    const int64_t value = source * 100000 + token;
                    require(value * (row[0] + 1) ==
                            (source * 100000 + inverse[source * config.num_tokens + token]) *
                                (row[0] + 1), "gather/scatter token value mapping");
                }
            }
        }
        if (static_cast<std::size_t>(ready) == publication.table_rows) {
            for (const auto& source : consumed) {
                require(std::all_of(source.begin(), source.end(), [](int n) { return n == 1; }),
                        "all source routes must be consumed");
            }
            require(std::all_of(packed_seen.begin(), packed_seen.end(), [](int n) { return n == 1; }),
                    "packed output must have no holes");
            require(state.expert_num_ready_tokens == std::vector<std::size_t>({0, 0}),
                    "unprocessed expert routes");
        }
    };

    for (uint64_t iteration = 0; iteration < 3; ++iteration) {
        buffers.begin_router_notification_iteration(iteration);
        std::fill(scatter.begin(), scatter.end(), -1);
        require(*static_cast<const int32_t*>(publication.device_ready_rows.ptr) == 0,
                "iteration resets published readiness");
        const std::array<int, 3> order = iteration % 2 == 0
            ? std::array<int, 3>{{0, 1, 2}} : std::array<int, 3>{{1, 0, 2}};
        for (const auto source : order) {
            buffers.process_router_notification_completion(0, source, iteration, 0, 1);
            check_publication();
            buffers.process_router_notification_completion(
                0, source, iteration, 1, config.num_tokens - 1);
            check_publication();
        }
        require(buffers.router_computation_scheduler_state().published_rows == 9,
                "all partial and full clusters published");
    }
}
}  // namespace

int main() {
    check_scheduler(9, 128, 2, 128, 256);
    check_scheduler(9, 256, 2, 256, 512);
    check_scheduler(10, 128, 2, 64, 128);
    check_scheduler(10, 256, 2, 128, 256);
    check_scheduler(10, 128, 1, 128, 128);
    check_scheduler(10, 64, 2, 64, 128);
    check_scheduler(10, 256, 4, 128, 512);
    ProxyConfig config;
    for (const int capacity : {0, 8, 11, 12}) {
        config.expert_gemm_device_capacity = capacity;
        bool rejected = false;
        try { (void)expert_gemm_cluster_rows(config); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "unsupported architecture must fail");
    }
    config.expert_gemm_device_capacity = 10;
    config.expert_gemm_m_tile = std::numeric_limits<std::size_t>::max();
    config.expert_gemm_cluster_m = 2;
    bool rejected = false;
    try { (void)expert_gemm_cluster_rows(config); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "cluster rows overflow must fail");
    std::cout << "test_gemm_geometry passed\n";
}
