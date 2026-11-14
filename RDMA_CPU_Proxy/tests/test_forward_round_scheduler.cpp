#include "cuda_buffers.hpp"
#include "logging.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace rdma_proxy;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template<class Action> static void rejects(Action action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "expected invalid completion/scheduling operation to fail");
}

struct alignas(16) SlabBlock { unsigned char bytes[16]; };

// This test exercises production scheduling and all native scatter policies in
// mock mode, even on a CUDA host. GPU scatter's input ABI is checked against an
// independent descriptor/reference walk; actual kernels require remote tests.
struct Fixture {
    ProxyConfig config;
    std::array<std::vector<int32_t>, 4> indices, offsets;
    std::array<std::vector<float>, 4> weights;
    std::vector<SlabBlock> slab;
    std::vector<int64_t> counts, inverse;
    std::array<uint64_t, 4> bases{{0x100000, 0x200000, 0x300000, 0x400000}};
    std::size_t routes{0}, cluster_rows{0}, weight_offset{0};
    int preparation;
    // Last member: stop/join scatter worker before borrowed slab destruction.
    std::unique_ptr<CudaBuffers> owner;

    Fixture(bool per_entry, bool weighted, int preparation_, int architecture,
            std::size_t tile = 2, bool memcpy_flag = false)
        : preparation(preparation_) {
        config.mock_mode = true;
        config.router_routing_enabled = true;
        config.router_weighted_contributions_enabled = weighted;
        config.nvlink_forwarding_enabled = true;
        config.nvlink_forward_completion_notifications_enabled = true;
        config.nvlink_forward_notification_flush_per_entry_enabled = per_entry;
        config.nvlink_forward_notification_flag_update_mode = memcpy_flag
            ? NvlinkForwardNotificationFlagUpdateMode::kMemcpy
            : NvlinkForwardNotificationFlagUpdateMode::kStreamWrite;
        config.num_nodes = 1;
        config.num_gpus_per_node = 4;
        config.router_num_experts = 8;
        config.router_top_k = 2;
        config.token_dimension = 256;
        config.expert_gemm_device_capacity = architecture;
        config.expert_gemm_m_tile = tile;
        config.expert_gemm_cluster_m = tile == 2 ? 1 : 2;
        config.expert_gemm_n_tile = 128;
        config.expert_gemm_dimension = 512;
        config.expert_gemm_max_swizzle_size = 2;
        cluster_rows = expert_gemm_cluster_rows(config);
        config.num_tokens = cluster_rows + 3;
        owner = std::make_unique<CudaBuffers>(config);
        owner->initialize();
        const std::array<std::array<std::size_t, 2>, 4> lengths{{
            {{cluster_rows + 1, cluster_rows}},
            {{cluster_rows + 2, cluster_rows - 1}},
            {{cluster_rows - 1, cluster_rows}}, {{0, 0}}}};
        for (int source = 0; source < 4; ++source) {
            offsets[source].push_back(0);
            for (int expert = 0; expert < 2; ++expert) {
                for (std::size_t token = 0; token < lengths[source][expert]; ++token) {
                    indices[source].push_back(static_cast<int32_t>(token));
                    weights[source].push_back(static_cast<float>(
                        1 + (token + source + expert) % 7) / 8.0f);
                }
                offsets[source].push_back(static_cast<int32_t>(indices[source].size()));
            }
            routes += indices[source].size();
            RouterExpertMetadata metadata;
            metadata.destination_node_rank = metadata.destination_gpu_index = 0;
            metadata.source_node_rank = 0;
            metadata.source_gpu_index = source;
            metadata.num_nodes = 1;
            metadata.num_gpus_per_node = 4;
            metadata.num_experts = 8;
            metadata.experts_per_gpu = 2;
            metadata.first_global_expert = 0;
            metadata.num_tokens = config.num_tokens;
            metadata.top_k = 2;
            metadata.weighted_contributions = weighted;
            metadata.expert_token_indices = indices[source];
            metadata.expert_offsets = offsets[source];
            if (weighted) metadata.expert_route_weights = weights[source];
            owner->install_expert_metadata(std::move(metadata));
        }
        if (preparation < 0) return;
        const auto scatter_bytes = routes * 2 * sizeof(int64_t);
        weight_offset = (scatter_bytes + 15) & ~std::size_t{15};
        const auto bytes = weighted ? weight_offset + routes * sizeof(float) : scatter_bytes;
        slab.resize((bytes + 15) / 16);
        counts.resize(4);
        inverse.resize(4 * config.num_tokens);
        const int32_t pairs[]{0, 0, 0, 1, 0, 2, 0, 3};
        RdmaProxyCpuScatterConfig setup{sizeof(setup), 4, routes,
            config.num_tokens, config.token_dimension, 128, 2,
            reinterpret_cast<uintptr_t>(slab.data()), pairs, bases.data(),
            counts.data(), inverse.data()};
        owner->configure_cpu_scatter(setup);
        if (weighted) owner->configure_cpu_route_weights(
            reinterpret_cast<uintptr_t>(slab.data()) + weight_offset, routes);
        if (preparation == 1) owner->configure_cpu_scatter_deferred(bytes);
        if (preparation == 2) owner->configure_cpu_scatter_threaded(bytes);
    }

    std::vector<int32_t> prefix() const {
        const auto& publication = owner->router_notification_publication_buffers();
        const auto count = owner->router_computation_scheduler_state().published_rows;
        const auto* table = static_cast<const int32_t*>(publication.device_table.ptr);
        return {table, table + count * publication.table_width};
    }

    void verify_final(uint64_t iteration) {
        if (preparation > 0) owner->wait_cpu_scatter_ready(iteration, 1, 5000);
        owner->synchronize_router_notification_publication();
        const auto& publication = owner->router_notification_publication_buffers();
        const auto state = owner->router_computation_scheduler_state();
        require(state.published_rows == publication.table_rows, "missing final descriptors");
        require(*static_cast<const int32_t*>(publication.device_ready_rows.ptr) ==
            static_cast<int32_t>(publication.table_rows), "ready prefix not published");
        require(std::memcmp(publication.host_table.ptr, publication.device_table.ptr,
            publication.host_table.bytes) == 0, "ready prefix preceded table copy");
        const auto table = prefix();
        std::array<std::vector<unsigned>, 4> seen;
        std::array<std::vector<double>, 4> expected, actual;
        for (int s = 0; s < 4; ++s) {
            seen[s].resize(indices[s].size());
            expected[s].resize(config.num_tokens);
            actual[s].resize(config.num_tokens);
            for (int e = 0; e < 2; ++e)
                for (int i = offsets[s][e]; i < offsets[s][e + 1]; ++i)
                    expected[s][indices[s][i]] += (s + indices[s][i] + 1) * (e + 2) *
                        (config.router_weighted_contributions_enabled ? weights[s][i] : 1.0f);
        }
        std::vector<unsigned> packed_seen(routes);
        const auto* scatter = reinterpret_cast<const int64_t*>(slab.data());
        const auto* packed_weights = preparation >= 0 ? reinterpret_cast<const float*>(
            reinterpret_cast<const unsigned char*>(slab.data()) + weight_offset) : nullptr;
        for (std::size_t r = 0; r < state.published_rows; r += publication.n_groups_per_m_cluster) {
            const auto* row = table.data() + r * publication.table_width;
            require(row[0] >= 0 && row[0] < 2 && row[1] == 0, "invalid expert/M bundle");
            for (std::size_t g = 0; g < publication.n_groups_per_m_cluster; ++g) {
                const auto* repeated = row + g * publication.table_width;
                require(repeated[0] == row[0] && repeated[1] == g * publication.group_size &&
                    std::equal(row + 2, row + publication.table_width, repeated + 2),
                    "N groups must repeat the same M bundle ranges");
            }
            std::size_t packed = 0, batch_rows = 0;
            for (int s = 0; s < 4; ++s) packed += row[2 + 2 * s];
            for (int s = 0; s < 4; ++s) {
                const auto begin = row[2 + 2 * s], end = row[3 + 2 * s];
                require(begin >= offsets[s][row[0]] && end >= begin &&
                    end <= offsets[s][row[0] + 1], "descriptor crosses expert range");
                const auto& buffer = owner->nvlink_receive_buffer_for_source(0, s);
                const auto* gpu_indices = static_cast<const int32_t*>(buffer.expert_token_indices_device.ptr);
                const auto* gpu_weights = static_cast<const float*>(buffer.expert_route_weights_device.ptr);
                for (int i = begin; i < end; ++i, ++packed, ++batch_rows) {
                    require(!seen[s][i]++ && packed < routes && !packed_seen[packed]++,
                        "route or packed slot duplicated");
                    const auto token = gpu_indices[i];
                    require(token == indices[s][i], "GPU scatter A_idx alignment");
                    const float weight = config.router_weighted_contributions_enabled ? gpu_weights[i] : 1.0f;
                    if (config.router_weighted_contributions_enabled)
                        require(weight == weights[s][i], "GPU scatter source weight alignment");
                    if (preparation >= 0) {
                        const auto first = inverse.begin() + s * config.num_tokens;
                        const auto compact = std::find(first, first + counts[s], token) - first;
                        require(compact < counts[s], "scatter destination missing");
                        require(scatter[2 * packed] == static_cast<int64_t>(bases[s] + compact * 256) &&
                            scatter[2 * packed + 1] == counts[s] * 256, "CPU scatter descriptor alignment");
                        if (config.router_weighted_contributions_enabled)
                            require(packed_weights[packed] == weight, "CPU packed route-weight alignment");
                    }
                    actual[s][token] += (s + token + 1) * (row[0] + 2) * weight;
                }
            }
            require(batch_rows > 0 && batch_rows <= cluster_rows, "architecture cluster capacity");
        }
        for (int s = 0; s < 4; ++s) {
            require(std::all_of(seen[s].begin(), seen[s].end(), [](unsigned n) { return n == 1; }),
                "not every route was scheduled exactly once");
            require(actual[s] == expected[s], "weighted gather/scatter reference composition");
            const auto progress = owner->expert_token_head_state_for_source(0, s);
            require(progress.expert_token_heads == progress.expert_token_tails, "unconsumed source readiness");
        }
        require(std::all_of(packed_seen.begin(), packed_seen.end(), [](unsigned n) { return n == 1; }),
            "packed output has holes");
    }
};

static RouterNotificationSchedulingDiagnostics run(bool aggregate, bool per_entry,
    bool weighted, int preparation, bool early, int architecture, std::size_t tile,
    bool memcpy_flag = false) {
    Fixture f(per_entry, weighted, preparation, architecture, tile, memcpy_flag);
    auto& buffers = *f.owner;
    for (uint64_t iteration = 0; iteration < 3; ++iteration) {
        buffers.begin_router_notification_iteration(iteration);
        const auto& publication = buffers.router_notification_publication_buffers();
        require(buffers.router_computation_scheduler_state().scheduling_passes == 0, "pass reset");
        rejects([&] { buffers.schedule_router_notification_completions(iteration + 1); });
        if (iteration) rejects([&] { buffers.ingest_router_notification_completion(0, 0, iteration - 1, 0, 0); });
        rejects([&] { buffers.ingest_router_notification_completion(0, 1, iteration, 1, 1); });
        if (early) {
            buffers.process_router_notification_completion(0, 0, iteration, 0, f.config.num_tokens);
            require(!f.prefix().empty(), "early-local input should schedule immediately");
        }
        const auto before = f.prefix();
        const auto first_flush = publication.table_flush_count;
        const auto first_ready = publication.ready_rows_publication_count;
        // This complete collection includes an empty source (zero data), with
        // uneven route counts and final partial clusters on other sources.
        std::array<int, 4> order{{0, 1, 2, 3}};
        if (iteration == 1) std::reverse(order.begin(), order.end());
        if (iteration == 2) std::rotate(order.begin(), order.begin() + 1, order.end());
        for (int source : order) {
            if (early && source == 0) continue; // own-source notifications suppressed
            if (aggregate) {
                buffers.ingest_router_notification_completion(0, source, iteration, 0,
                    source == 3 ? 0 : f.config.num_tokens);
                require(f.prefix() == before && publication.table_flush_count == first_flush &&
                    publication.ready_rows_publication_count == first_ready,
                    "ingestion scheduled or published before complete collection");
            } else {
                buffers.process_router_notification_completion(0, source, iteration, 0,
                    source == 3 ? 0 : f.config.num_tokens);
            }
        }
        if (aggregate) buffers.schedule_router_notification_completions(iteration);
        const auto after = f.prefix();
        require(std::equal(before.begin(), before.end(), after.begin()), "published prefix overwritten");
        if (aggregate) {
            const auto rows = (after.size() - before.size()) / publication.table_width;
            require(publication.table_flush_count - first_flush == (per_entry ? rows : 1),
                "collection publication policy changed");
            require(publication.ready_rows_publication_count - first_ready == (per_entry ? rows : 1),
                "collection ready-prefix publication count");
            int32_t expert = -1;
            for (std::size_t i = before.size(); i < after.size(); i += publication.table_width) {
                require(after[i] >= expert, "collection descriptors not grouped by expert");
                expert = after[i];
            }
        }
        f.verify_final(iteration);
        const auto final_flushes = publication.table_flush_count;
        buffers.schedule_router_notification_completions(iteration);
        require(publication.table_flush_count == final_flushes, "empty scheduling pass republished work");
    }
    return buffers.router_notification_scheduling_diagnostics();
}

static void partial_collections() {
    Fixture f(false, true, 2, 9);
    auto& buffers = *f.owner;
    buffers.begin_router_notification_iteration(0);
    for (int source : {2, 0, 1}) buffers.ingest_router_notification_completion(0, source, 0, 0, 1);
    require(f.prefix().empty(), "partial collection published before scheduler release");
    buffers.schedule_router_notification_completions(0);
    const auto first = f.prefix();
    require(!first.empty(), "combined partial source readiness failed to form clusters");
    rejects([&] { buffers.ingest_router_notification_completion(0, 0, 0, 0, 1); });
    for (int source : {1, 2, 0}) {
        buffers.ingest_router_notification_completion(0, source, 0, 1, f.config.num_tokens - 1);
        require(f.prefix() == first, "later partial collection changed ready prefix");
    }
    buffers.schedule_router_notification_completions(0);
    f.verify_final(0);
}

int main() {
    try {
        Logger::instance().set_level(LogLevel::kWarn);
        const auto immediate = run(false, false, true, -1, false, 9, 2);
        const auto collected = run(true, false, true, -1, false, 9, 2);
        require(collected.expert_runs == 2 && collected.expert_runs < immediate.expert_runs &&
            collected.max_consecutive_m_clusters > immediate.max_consecutive_m_clusters,
            "collection did not lengthen expert runs");
        require(collected.scheduling_passes < immediate.scheduling_passes, "collection scheduling count");
        for (bool per_entry : {false, true}) for (bool weighted : {false, true})
            for (int preparation : {-1, 0, 1, 2}) for (bool early : {false, true})
                run(true, per_entry, weighted, preparation, early, 9, 2, per_entry);
        for (int architecture : {9, 10}) for (std::size_t tile : {128, 256})
            for (int preparation : {-1, 0, 1, 2})
                run(true, true, true, preparation, true, architecture, tile);
        partial_collections();
        std::cout << "forward round scheduler tests passed (CPU/mock only)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
