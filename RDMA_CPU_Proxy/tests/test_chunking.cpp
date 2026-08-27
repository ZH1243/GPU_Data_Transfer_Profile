#include "cuda_buffers.hpp"
#include "protocol.hpp"
#include "qp_worker.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

int main() {
    using namespace rdma_proxy;

    const auto chunks = compute_chunks(
        /*num_tokens=*/5000,
        /*token_dimension=*/128,
        /*dtype_size=*/2,
        /*tokens_per_chunk=*/32,
        /*num_qps_per_peer=*/10);

    assert(chunks.size() == 157);
    assert(chunks[0].chunk_index == 0);
    assert(chunks[0].start_token == 0);
    assert(chunks[0].num_tokens == 32);
    assert(chunks[0].src_offset_bytes == 0);
    assert(chunks[0].dst_offset_bytes == 0);
    assert(chunks[0].length_bytes == 32 * 128 * 2);
    assert(chunks[0].qp_index == -1);
    assert(decode_immediate(chunks[0].imm_data) == 0);

    assert(chunks.back().chunk_index == 156);
    assert(chunks.back().start_token == 4992);
    assert(chunks.back().num_tokens == 8);
    assert(chunks.back().length_bytes == 8 * 128 * 2);
    assert(chunks.back().source_token_indices.empty());

    const auto discontinuous_chunks = compute_chunks(
        /*num_tokens=*/10,
        /*token_dimension=*/4,
        /*dtype_size=*/2,
        /*tokens_per_chunk=*/3,
        /*num_qps_per_peer=*/2,
        /*discontinuous_token_payload=*/true);
    assert(discontinuous_chunks.size() == 4);
    std::vector<std::size_t> assigned;
    for (const auto& chunk : discontinuous_chunks) {
        assert(chunk.source_token_indices.size() == chunk.num_tokens);
        assigned.insert(
            assigned.end(),
            chunk.source_token_indices.begin(),
            chunk.source_token_indices.end());
    }
    auto sorted = assigned;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        assert(sorted[i] == i);
    }
    bool differs_from_contiguous = false;
    for (std::size_t i = 0; i < assigned.size(); ++i) {
        differs_from_contiguous = differs_from_contiguous || assigned[i] != i;
    }
    assert(differs_from_contiguous);

    const std::vector<std::size_t> router_order{3, 1, 4, 9, 10, 45};
    const auto router_chunks = compute_chunks_from_token_indices(
        router_order,
        /*token_dimension=*/4,
        /*dtype_size=*/2,
        /*tokens_per_chunk=*/3,
        /*num_qps_per_peer=*/2);
    assert(router_chunks.size() == 2);
    assert((router_chunks[0].source_token_indices == std::vector<std::size_t>{3, 1, 4}));
    assert((router_chunks[1].source_token_indices == std::vector<std::size_t>{9, 10, 45}));
    assert(router_chunks[0].dst_offset_bytes == 0);
    assert(router_chunks[1].dst_offset_bytes == 3 * 4 * 2);
    assert(router_chunks[0].length_bytes == 3 * 4 * 2);

    RouterX3Metadata metadata;
    metadata.source_node_rank = 2;
    metadata.destination_node_rank = 0;
    metadata.local_gpu_index = 3;
    metadata.num_nodes = 4;
    metadata.num_gpus_per_node = 8;
    metadata.num_experts = 256;
    metadata.top_k = 8;
    metadata.num_tokens = 64;
    metadata.token_dimension = 4;
    metadata.element_bytes = 2;
    metadata.tokens_per_chunk = 3;
    metadata.token_indices = router_order;
    metadata.token_masks = {0x81, 0x42, 0x24, 0x18, 0x08, 0x01};
    const auto decoded_metadata = deserialize_router_x3_metadata(
        serialize_router_x3_metadata(metadata), metadata.num_tokens);
    assert(decoded_metadata.source_node_rank == metadata.source_node_rank);
    assert(decoded_metadata.destination_node_rank == metadata.destination_node_rank);
    assert(decoded_metadata.local_gpu_index == metadata.local_gpu_index);
    assert(decoded_metadata.token_indices == router_order);
    assert(decoded_metadata.token_masks == metadata.token_masks);
    const auto normalized_x4 = normalize_router_x4_for_nvlink(
        std::vector<uint8_t>{0x9, 0x6, 0x1}, 4);
    assert((normalized_x4 == std::vector<uint8_t>{0x90, 0x60, 0x10}));
    bool rejected_invalid_x4 = false;
    try {
        (void)normalize_router_x4_for_nvlink(std::vector<uint8_t>{0x10}, 4);
    } catch (const std::runtime_error&) {
        rejected_invalid_x4 = true;
    }
    assert(rejected_invalid_x4);

    RouterExpertMetadata expert_metadata;
    expert_metadata.source_node_rank = 2;
    expert_metadata.source_gpu_index = 3;
    expert_metadata.destination_node_rank = 0;
    expert_metadata.destination_gpu_index = 1;
    expert_metadata.num_nodes = 4;
    expert_metadata.num_gpus_per_node = 8;
    expert_metadata.num_experts = 256;
    expert_metadata.experts_per_gpu = 8;
    expert_metadata.first_global_expert = 8;
    expert_metadata.num_tokens = 64;
    expert_metadata.expert_token_indices = {0, 3, 1, 4, 7};
    expert_metadata.expert_offsets = {0, 2, 2, 3, 3, 5, 5, 5, 5};
    const auto decoded_expert_metadata = deserialize_router_expert_metadata(
        serialize_router_expert_metadata(expert_metadata),
        expert_metadata.num_tokens,
        metadata.num_tokens * static_cast<std::size_t>(metadata.top_k));
    assert(decoded_expert_metadata.source_node_rank == 2);
    assert(decoded_expert_metadata.source_gpu_index == 3);
    assert(decoded_expert_metadata.destination_gpu_index == 1);
    assert(decoded_expert_metadata.first_global_expert == 8);
    assert(decoded_expert_metadata.expert_token_indices ==
           expert_metadata.expert_token_indices);
    assert(decoded_expert_metadata.expert_offsets == expert_metadata.expert_offsets);
    assert(decode_immediate(encode_immediate(1234)) == 1234);

    PeerConnectionInfo info;
    info.node_rank = 2;
    info.gpu_index = 0;
    info.recv_buffer = MemoryRegionInfo{0x1234, 11, 22, 4096};
    QPInfo qp;
    qp.qp_num = 7;
    qp.lid = 3;
    qp.psn = 99;
    qp.gid[15] = 42;
    info.qps.push_back(qp);
    const auto decoded = deserialize_peer_info(serialize_peer_info(info));
    assert(decoded.node_rank == info.node_rank);
    assert(decoded.gpu_index == info.gpu_index);
    assert(decoded.recv_buffer.addr == info.recv_buffer.addr);
    assert(decoded.recv_buffer.lkey == info.recv_buffer.lkey);
    assert(decoded.recv_buffer.rkey == info.recv_buffer.rkey);
    assert(decoded.qps.size() == 1);
    assert(decoded.qps[0].gid[15] == 42);

    ProxyConfig router_nvlink_config;
    router_nvlink_config.mock_mode = true;
    router_nvlink_config.router_routing_enabled = true;
    router_nvlink_config.nvlink_forwarding_enabled = true;
    router_nvlink_config.nvlink_forward_completion_notifications_enabled = true;
    router_nvlink_config.nvlink_forward_notification_flush_only_enabled = true;
    router_nvlink_config.nvlink_forward_notification_flag_update_mode =
        NvlinkForwardNotificationFlagUpdateMode::kMemcpy;
    router_nvlink_config.router_computation_forwarded_inputs_only = true;
    router_nvlink_config.router_num_experts = 256;
    router_nvlink_config.router_top_k = 8;
    router_nvlink_config.node_rank = 0;
    router_nvlink_config.num_nodes = 4;
    router_nvlink_config.local_gpu_index = 3;
    router_nvlink_config.num_gpus_per_node = 8;
    router_nvlink_config.num_tokens = 4;
    router_nvlink_config.token_dimension = 2;
    router_nvlink_config.dtype = DataType::kFP16;
    router_nvlink_config.peers = {
        PeerAddress{1, "peer-1", 1},
        PeerAddress{2, "peer-2", 1},
        PeerAddress{3, "peer-3", 1},
    };
    CudaBuffers router_nvlink_buffers(router_nvlink_config);
    router_nvlink_buffers.initialize();
    assert(router_nvlink_buffers.nvlink_receive_buffers().size() == 32);
    const auto expected_forwarding_buffer_bytes = router_nvlink_buffers.token_buffer_bytes();
    assert(router_nvlink_buffers.nvlink_receive_buffer_bytes() == expected_forwarding_buffer_bytes);
    std::set<void*> distinct_router_nvlink_allocations;
    for (int source_node = 0; source_node < router_nvlink_config.num_nodes; ++source_node) {
        for (int source_gpu = 0; source_gpu < router_nvlink_config.num_gpus_per_node; ++source_gpu) {
            const auto& buffer = router_nvlink_buffers.nvlink_receive_buffer_for_source(
                source_node, source_gpu);
            assert(buffer.source_node_rank == source_node);
            assert(buffer.source_gpu_index == source_gpu);
            assert(buffer.recv.ptr != nullptr);
            assert(buffer.recv.bytes == expected_forwarding_buffer_bytes);
            assert(!buffer.expert_metadata_ready);
            if (source_node != router_nvlink_config.node_rank &&
                source_gpu == router_nvlink_config.local_gpu_index) {
                assert(buffer.recv.ptr ==
                       router_nvlink_buffers.buffers_for_peer(source_node).recv.ptr);
                assert(buffer.recv.is_alias);
            } else {
                assert(!buffer.recv.is_alias);
            }
            distinct_router_nvlink_allocations.insert(buffer.recv.ptr);
        }
    }
    assert(distinct_router_nvlink_allocations.size() == 32);

    auto local_expert_metadata = expert_metadata;
    local_expert_metadata.source_node_rank = 2;
    local_expert_metadata.source_gpu_index = 3;
    local_expert_metadata.destination_node_rank = router_nvlink_config.node_rank;
    local_expert_metadata.destination_gpu_index = router_nvlink_config.local_gpu_index;
    local_expert_metadata.num_experts = router_nvlink_config.router_num_experts;
    local_expert_metadata.experts_per_gpu = 8;
    local_expert_metadata.first_global_expert = 24;
    local_expert_metadata.num_tokens = router_nvlink_config.num_tokens;
    local_expert_metadata.expert_token_indices = {0, 1, 2};
    local_expert_metadata.expert_offsets = {0, 1, 1, 2, 2, 3, 3, 3, 3};
    for (int source_node = 0; source_node < router_nvlink_config.num_nodes; ++source_node) {
        for (int source_gpu = 0;
             source_gpu < router_nvlink_config.num_gpus_per_node; ++source_gpu) {
            auto source_metadata = local_expert_metadata;
            source_metadata.source_node_rank = source_node;
            source_metadata.source_gpu_index = source_gpu;
            if (source_node != 2 || source_gpu != 3) {
                source_metadata.expert_token_indices.clear();
                source_metadata.expert_offsets.assign(9, 0);
            }
            router_nvlink_buffers.install_expert_metadata(
                std::move(source_metadata));
        }
    }

    const auto& publication =
        router_nvlink_buffers.router_notification_publication_buffers();
    const std::size_t expected_remote_inputs = 3 * 8;
    const std::size_t expected_table_width = 2 + 2 * expected_remote_inputs;
    const std::size_t expected_table_row_bytes =
        expected_table_width * sizeof(int32_t);
    assert(publication.table_width == expected_table_width);
    assert(publication.table_row_bytes == expected_table_row_bytes);
    assert(publication.table_rows == 12);
    assert(publication.num_input_buffers == expected_remote_inputs);
    assert(publication.a_idx_capacity == 3);
    assert(publication.n_groups_per_m_cluster == 4);
    assert(publication.group_size == 8);
    assert(publication.host_table.ptr != nullptr);
    assert(publication.host_table.bytes == 12 * expected_table_row_bytes);
    assert(publication.host_ready_rows.ptr != nullptr);
    assert(publication.host_ready_rows.bytes == sizeof(int32_t));
    assert(publication.device_table.ptr != nullptr);
    assert(publication.device_table.bytes == publication.host_table.bytes);
    assert(publication.device_ready_rows.ptr != nullptr);
    assert(publication.device_ready_rows.bytes == sizeof(int32_t));
    assert(std::all_of(
        static_cast<const uint8_t*>(publication.host_table.ptr),
        static_cast<const uint8_t*>(publication.host_table.ptr) + publication.host_table.bytes,
        [](uint8_t value) { return value == 0; }));
    assert(*static_cast<const int32_t*>(publication.host_ready_rows.ptr) == 0);
    std::memset(publication.host_table.ptr, 0x5a, publication.host_table.bytes);
    *static_cast<int32_t*>(publication.host_ready_rows.ptr) = 7;
    std::memset(publication.device_table.ptr, 0xff, publication.device_table.bytes);
    *static_cast<int32_t*>(publication.device_ready_rows.ptr) = 3;
    router_nvlink_buffers.flush_router_notification_publication();
    assert(std::all_of(
        static_cast<const uint8_t*>(publication.device_table.ptr),
        static_cast<const uint8_t*>(publication.device_table.ptr) + publication.device_table.bytes,
        [](uint8_t value) { return value == 0x5a; }));
    assert(*static_cast<const int32_t*>(publication.device_ready_rows.ptr) == 7);
    std::memset(publication.host_table.ptr, 0, publication.host_table.bytes);
    *static_cast<int32_t*>(publication.host_ready_rows.ptr) = 0;
    const auto& installed = router_nvlink_buffers.nvlink_receive_buffer_for_source(2, 3);
    assert(installed.expert_metadata_ready);
    assert(installed.expert_metadata.expert_token_indices ==
           local_expert_metadata.expert_token_indices);
    assert(installed.expert_token_index_count == 3);
    assert(installed.expert_token_indices_device.bytes == 3 * sizeof(int32_t));
    assert((std::vector<int32_t>(
                static_cast<const int32_t*>(installed.expert_token_indices_device.ptr),
                static_cast<const int32_t*>(installed.expert_token_indices_device.ptr) + 3) ==
            std::vector<int32_t>{0, 1, 2}));
    const auto& empty_installed =
        router_nvlink_buffers.nvlink_receive_buffer_for_source(1, 0);
    assert(empty_installed.expert_token_index_count == 0);
    assert(empty_installed.expert_token_indices_device.bytes == 3 * sizeof(int32_t));
    assert((std::vector<int32_t>(
                static_cast<const int32_t*>(empty_installed.expert_token_indices_device.ptr),
                static_cast<const int32_t*>(empty_installed.expert_token_indices_device.ptr) + 3) ==
            std::vector<int32_t>{0, 0, 0}));
    assert(router_nvlink_buffers.nvlink_receive_buffer_for_source(
               0, 0).expert_token_indices_device.ptr == nullptr);
    auto head_state = router_nvlink_buffers.expert_token_head_state_for_source(2, 3);
    assert(!head_state.iteration_initialized);
    assert(head_state.received_token_frontier == 0);
    assert(head_state.expert_token_heads == std::vector<std::size_t>(8, 0));
    assert(head_state.expert_token_tails == std::vector<std::size_t>(8, 0));

    // Flush-only mode publishes the first table row and current ready-row flag
    // for a completion without touching
    // the source buffer's iteration, frontier, or expert Heads.
    *static_cast<int32_t*>(publication.host_ready_rows.ptr) = 11;
    *static_cast<int32_t*>(publication.device_ready_rows.ptr) = 3;
    router_nvlink_buffers.process_router_notification_completion(2, 3, 0, 0, 1);
    head_state = router_nvlink_buffers.expert_token_head_state_for_source(2, 3);
    assert(!head_state.iteration_initialized);
    assert(head_state.received_token_frontier == 0);
    assert(head_state.expert_token_heads == std::vector<std::size_t>(8, 0));
    assert(head_state.expert_token_tails == std::vector<std::size_t>(8, 0));
    assert(*static_cast<const int32_t*>(publication.device_ready_rows.ptr) == 11);
    *static_cast<int32_t*>(publication.host_ready_rows.ptr) = 0;

    router_nvlink_buffers.update_expert_token_heads(2, 3, 0, 0, 1);
    head_state = router_nvlink_buffers.expert_token_head_state_for_source(2, 3);
    assert(head_state.iteration_initialized);
    assert(head_state.iteration == 0);
    assert(head_state.received_token_frontier == 1);
    assert((head_state.expert_token_heads ==
            std::vector<std::size_t>{1, 0, 0, 0, 0, 0, 0, 0}));

    router_nvlink_buffers.update_expert_token_heads(2, 3, 0, 1, 2);
    head_state = router_nvlink_buffers.expert_token_head_state_for_source(2, 3);
    assert(head_state.received_token_frontier == 3);
    assert((head_state.expert_token_heads ==
            std::vector<std::size_t>{1, 0, 1, 0, 1, 0, 0, 0}));

    // The first notification for the next iteration resets every Head before
    // advancing it against that iteration's received prefix.
    router_nvlink_buffers.update_expert_token_heads(2, 3, 1, 0, 2);
    head_state = router_nvlink_buffers.expert_token_head_state_for_source(2, 3);
    assert(head_state.iteration == 1);
    assert(head_state.received_token_frontier == 2);
    assert((head_state.expert_token_heads ==
            std::vector<std::size_t>{1, 0, 1, 0, 0, 0, 0, 0}));

    bool rejected_noncontiguous_notification = false;
    try {
        router_nvlink_buffers.update_expert_token_heads(2, 3, 1, 3, 1);
    } catch (const std::runtime_error&) {
        rejected_noncontiguous_notification = true;
    }
    assert(rejected_noncontiguous_notification);

    // Build three M-clusters for one local expert from two source-specific
    // receive buffers. Each cluster expands to two QuACK N-group rows. The
    // final cluster has one token, so it must be emitted without waiting for
    // the configured four-token M-cluster capacity.
    ProxyConfig scheduler_config;
    scheduler_config.mock_mode = true;
    scheduler_config.router_routing_enabled = true;
    scheduler_config.nvlink_forwarding_enabled = true;
    scheduler_config.nvlink_forward_completion_notifications_enabled = true;
    scheduler_config.router_num_experts = 2;
    scheduler_config.router_top_k = 1;
    scheduler_config.node_rank = 0;
    scheduler_config.num_nodes = 1;
    scheduler_config.local_gpu_index = 0;
    scheduler_config.num_gpus_per_node = 2;
    scheduler_config.num_tokens = 7;
    scheduler_config.token_dimension = 2;
    scheduler_config.dtype = DataType::kFP16;
    scheduler_config.expert_gemm_m_tile = 4;
    scheduler_config.expert_gemm_n_tile = 2;
    scheduler_config.expert_gemm_dimension = 8;
    scheduler_config.expert_gemm_cluster_m = 1;
    scheduler_config.expert_gemm_max_swizzle_size = 2;
    CudaBuffers scheduler_buffers(scheduler_config);
    scheduler_buffers.initialize();

    RouterExpertMetadata scheduler_metadata;
    scheduler_metadata.destination_node_rank = 0;
    scheduler_metadata.destination_gpu_index = 0;
    scheduler_metadata.num_nodes = 1;
    scheduler_metadata.num_gpus_per_node = 2;
    scheduler_metadata.num_experts = 2;
    scheduler_metadata.experts_per_gpu = 1;
    scheduler_metadata.first_global_expert = 0;
    scheduler_metadata.num_tokens = 7;
    scheduler_metadata.source_node_rank = 0;
    scheduler_metadata.source_gpu_index = 0;
    scheduler_metadata.expert_token_indices = {0, 1, 4};
    scheduler_metadata.expert_offsets = {0, 3};
    scheduler_buffers.install_expert_metadata(scheduler_metadata);
    scheduler_metadata.source_gpu_index = 1;
    scheduler_metadata.expert_token_indices = {0, 2, 3, 4, 5, 6};
    scheduler_metadata.expert_offsets = {0, 6};
    scheduler_buffers.install_expert_metadata(scheduler_metadata);

    const auto& scheduler_publication =
        scheduler_buffers.router_notification_publication_buffers();
    assert(scheduler_publication.table_width == 6);
    assert(scheduler_publication.table_row_bytes == 6 * sizeof(int32_t));
    assert(scheduler_publication.table_rows == 6);
    assert(scheduler_publication.num_input_buffers == 2);
    assert(scheduler_publication.a_idx_capacity == 6);
    assert(scheduler_publication.n_groups_per_m_cluster == 2);
    assert(scheduler_publication.group_size == 2);
    assert(scheduler_publication.host_table.bytes == 36 * sizeof(int32_t));
    const auto& scheduler_source0 =
        scheduler_buffers.nvlink_receive_buffer_for_source(0, 0);
    const auto& scheduler_source1 =
        scheduler_buffers.nvlink_receive_buffer_for_source(0, 1);
    assert(scheduler_source0.expert_token_index_count == 3);
    assert(scheduler_source1.expert_token_index_count == 6);
    assert((std::vector<int32_t>(
                static_cast<const int32_t*>(
                    scheduler_source0.expert_token_indices_device.ptr),
                static_cast<const int32_t*>(
                    scheduler_source0.expert_token_indices_device.ptr) + 6) ==
            std::vector<int32_t>{0, 1, 4, 0, 0, 0}));
    assert((std::vector<int32_t>(
                static_cast<const int32_t*>(
                    scheduler_source1.expert_token_indices_device.ptr),
                static_cast<const int32_t*>(
                    scheduler_source1.expert_token_indices_device.ptr) + 6) ==
            std::vector<int32_t>{0, 2, 3, 4, 5, 6}));

    scheduler_buffers.begin_router_notification_iteration(0);
    std::memset(
        scheduler_publication.device_table.ptr, 0xff,
        scheduler_publication.device_table.bytes);

    scheduler_buffers.process_router_notification_completion(0, 0, 0, 0, 2);
    assert(*static_cast<const int32_t*>(
               scheduler_publication.device_ready_rows.ptr) == 0);
    assert(std::all_of(
        static_cast<const uint8_t*>(scheduler_publication.device_table.ptr),
        static_cast<const uint8_t*>(scheduler_publication.device_table.ptr) +
            scheduler_publication.device_table.bytes,
        [](uint8_t value) { return value == 0xff; }));

    scheduler_buffers.process_router_notification_completion(0, 1, 0, 0, 3);
    assert(*static_cast<const int32_t*>(
               scheduler_publication.device_ready_rows.ptr) == 2);
    const auto* table = static_cast<const int32_t*>(
        scheduler_publication.device_table.ptr);
    const auto row = [&](std::size_t index) {
        return table + index * scheduler_publication.table_width;
    };
    assert((std::vector<int32_t>(row(0), row(0) + 6) ==
            std::vector<int32_t>{0, 0, 0, 2, 0, 2}));
    assert((std::vector<int32_t>(row(1), row(1) + 6) ==
            std::vector<int32_t>{0, 2, 0, 2, 0, 2}));
    assert(std::all_of(
        reinterpret_cast<const uint8_t*>(row(2)),
        static_cast<const uint8_t*>(scheduler_publication.device_table.ptr) +
            scheduler_publication.device_table.bytes,
        [](uint8_t value) { return value == 0xff; }));

    scheduler_buffers.process_router_notification_completion(0, 0, 0, 2, 3);
    assert(*static_cast<const int32_t*>(
               scheduler_publication.device_ready_rows.ptr) == 2);
    scheduler_buffers.process_router_notification_completion(0, 1, 0, 3, 3);
    assert(*static_cast<const int32_t*>(
               scheduler_publication.device_ready_rows.ptr) == 4);
    assert((std::vector<int32_t>(row(2), row(2) + 6) ==
            std::vector<int32_t>{0, 0, 2, 3, 2, 5}));
    assert((std::vector<int32_t>(row(3), row(3) + 6) ==
            std::vector<int32_t>{0, 2, 2, 3, 2, 5}));

    scheduler_buffers.process_router_notification_completion(0, 1, 0, 6, 1);
    assert(*static_cast<const int32_t*>(
               scheduler_publication.device_ready_rows.ptr) == 6);
    assert((std::vector<int32_t>(row(4), row(4) + 6) ==
            std::vector<int32_t>{0, 0, 3, 3, 5, 6}));
    assert((std::vector<int32_t>(row(5), row(5) + 6) ==
            std::vector<int32_t>{0, 2, 3, 3, 5, 6}));

    const auto scheduler_state =
        scheduler_buffers.router_computation_scheduler_state();
    assert(scheduler_state.initialized);
    assert(scheduler_state.published_batches == 3);
    assert(scheduler_state.published_rows == 6);
    assert(scheduler_state.expert_total_tokens == std::vector<std::size_t>{9});
    assert(scheduler_state.expert_num_ready_tokens == std::vector<std::size_t>{0});
    assert(scheduler_state.expert_num_notified_batches == std::vector<std::size_t>{3});
    assert(scheduler_state.expert_total_batches == std::vector<std::size_t>{3});
    const auto source0_progress =
        scheduler_buffers.expert_token_head_state_for_source(0, 0);
    const auto source1_progress =
        scheduler_buffers.expert_token_head_state_for_source(0, 1);
    assert(source0_progress.expert_token_heads == std::vector<std::size_t>{3});
    assert(source0_progress.expert_token_tails == std::vector<std::size_t>{3});
    assert(source1_progress.expert_token_heads == std::vector<std::size_t>{6});
    assert(source1_progress.expert_token_tails == std::vector<std::size_t>{6});

    // A new iteration resets the GPU-visible ready-row prefix even when its
    // first completion has too few tokens to produce a table row.
    scheduler_buffers.begin_router_notification_iteration(1);
    assert(*static_cast<const int32_t*>(
               scheduler_publication.device_ready_rows.ptr) == 0);
    scheduler_buffers.process_router_notification_completion(0, 0, 1, 0, 1);
    assert(*static_cast<const int32_t*>(
               scheduler_publication.device_ready_rows.ptr) == 0);
    const auto reset_scheduler_state =
        scheduler_buffers.router_computation_scheduler_state();
    assert(reset_scheduler_state.iteration == 1);
    assert(reset_scheduler_state.published_batches == 0);
    assert(reset_scheduler_state.published_rows == 0);
    assert(reset_scheduler_state.expert_num_ready_tokens ==
           std::vector<std::size_t>{1});
    assert(reset_scheduler_state.expert_num_notified_batches ==
           std::vector<std::size_t>{0});
    const auto reset_source1_progress =
        scheduler_buffers.expert_token_head_state_for_source(0, 1);
    assert(!reset_source1_progress.iteration_initialized);
    assert(reset_source1_progress.expert_token_heads ==
           std::vector<std::size_t>{0});
    assert(reset_source1_progress.expert_token_tails ==
           std::vector<std::size_t>{0});

    // With all nine expert tokens becoming ready on the second notification,
    // the default policy coalesces six appended rows into one publication.
    // Per-entry mode publishes those same six rows independently.
    const auto run_flush_policy = [&](bool flush_per_entry) {
        auto policy_config = scheduler_config;
        policy_config.nvlink_forward_notification_flush_per_entry_enabled =
            flush_per_entry;
        CudaBuffers policy_buffers(policy_config);
        policy_buffers.initialize();

        auto source0_metadata = scheduler_metadata;
        source0_metadata.source_gpu_index = 0;
        source0_metadata.expert_token_indices = {0, 1, 4};
        source0_metadata.expert_offsets = {0, 3};
        policy_buffers.install_expert_metadata(source0_metadata);
        auto source1_metadata = scheduler_metadata;
        source1_metadata.source_gpu_index = 1;
        source1_metadata.expert_token_indices = {0, 2, 3, 4, 5, 6};
        source1_metadata.expert_offsets = {0, 6};
        policy_buffers.install_expert_metadata(source1_metadata);

        const auto& policy_publication =
            policy_buffers.router_notification_publication_buffers();
        policy_buffers.begin_router_notification_iteration(0);
        policy_buffers.process_router_notification_completion(0, 0, 0, 0, 7);
        const auto table_flushes_before = policy_publication.table_flush_count;
        const auto ready_publications_before =
            policy_publication.ready_rows_publication_count;
        policy_buffers.process_router_notification_completion(0, 1, 0, 0, 7);
        assert(*static_cast<const int32_t*>(
                   policy_publication.device_ready_rows.ptr) == 6);
        const auto policy_state =
            policy_buffers.router_computation_scheduler_state();
        assert(policy_state.published_batches == 3);
        assert(policy_state.published_rows == 6);
        return std::make_pair(
            policy_publication.table_flush_count - table_flushes_before,
            policy_publication.ready_rows_publication_count -
                ready_publications_before);
    };
    assert(run_flush_policy(false) ==
           std::make_pair(uint64_t{1}, uint64_t{1}));
    assert(run_flush_policy(true) ==
           std::make_pair(uint64_t{6}, uint64_t{6}));

    // The embedding allocator receives both the statically shaped transport
    // buffers and the router-dependent QuACK tensors allocated after metadata
    // installation. CudaBuffers borrows every returned pointer.
    std::vector<void*> external_device_pointers;
    std::vector<DeviceBufferAllocationRequest> external_device_requests;
    {
        CudaBuffers external_buffers(scheduler_config);
        external_buffers.set_external_device_buffer_allocator(
            [&](const DeviceBufferAllocationRequest& request) -> void* {
                void* pointer = ::operator new(request.bytes);
                std::memset(pointer, 0, request.bytes);
                external_device_pointers.push_back(pointer);
                external_device_requests.push_back(request);
                return pointer;
            });
        external_buffers.initialize();
        auto source0_metadata = scheduler_metadata;
        source0_metadata.source_gpu_index = 0;
        source0_metadata.expert_token_indices = {0, 1, 4};
        source0_metadata.expert_offsets = {0, 3};
        external_buffers.install_expert_metadata(source0_metadata);
        auto source1_metadata = scheduler_metadata;
        source1_metadata.source_gpu_index = 1;
        source1_metadata.expert_token_indices = {0, 2, 3, 4, 5, 6};
        source1_metadata.expert_offsets = {0, 6};
        external_buffers.install_expert_metadata(source1_metadata);

        const auto count_kind = [&](DeviceBufferKind kind) {
            return std::count_if(
                external_device_requests.begin(),
                external_device_requests.end(),
                [&](const DeviceBufferAllocationRequest& request) {
                    return request.kind == kind;
                });
        };
        assert(external_device_requests.size() == 7);
        assert(count_kind(DeviceBufferKind::kRdmaSend) == 1);
        assert(count_kind(DeviceBufferKind::kNvlinkReceive) == 2);
        assert(count_kind(DeviceBufferKind::kRouterAIdx) == 2);
        assert(count_kind(DeviceBufferKind::kGatherTable) == 1);
        assert(count_kind(DeviceBufferKind::kGatherReadyRows) == 1);
        const auto& last_request = external_device_requests.back();
        assert(last_request.kind == DeviceBufferKind::kGatherTable);
        assert((last_request.dimensions == std::vector<std::size_t>{6, 6}));
    }
    for (void* pointer : external_device_pointers) ::operator delete(pointer);

    // The remote-input GroupedGEMM scope includes both the compact forwarded
    // input (1, 1) and the direct RDMA alias (1, 0), while excluding local-node
    // buffers.
    auto forwarded_config = scheduler_config;
    forwarded_config.num_nodes = 2;
    forwarded_config.num_gpus_per_node = 2;
    forwarded_config.router_num_experts = 4;
    forwarded_config.router_computation_forwarded_inputs_only = true;
    forwarded_config.peers = {PeerAddress{1, "peer-1", 1}};
    CudaBuffers forwarded_buffers(forwarded_config);
    forwarded_buffers.initialize();
    for (int source_node = 0; source_node < 2; ++source_node) {
        for (int source_gpu = 0; source_gpu < 2; ++source_gpu) {
            RouterExpertMetadata metadata;
            metadata.source_node_rank = source_node;
            metadata.source_gpu_index = source_gpu;
            metadata.destination_node_rank = 0;
            metadata.destination_gpu_index = 0;
            metadata.num_nodes = 2;
            metadata.num_gpus_per_node = 2;
            metadata.num_experts = 4;
            metadata.experts_per_gpu = 1;
            metadata.first_global_expert = 0;
            metadata.num_tokens = 7;
            metadata.expert_token_indices = source_node == 1
                ? (source_gpu == 0
                    ? std::vector<int32_t>{0, 2}
                    : std::vector<int32_t>{0, 1, 2})
                : std::vector<int32_t>{};
            metadata.expert_offsets = {
                0, static_cast<int32_t>(metadata.expert_token_indices.size())};
            forwarded_buffers.install_expert_metadata(std::move(metadata));
        }
    }
    const auto& forwarded_publication =
        forwarded_buffers.router_notification_publication_buffers();
    assert(forwarded_publication.num_input_buffers == 2);
    assert(forwarded_publication.table_width == 6);
    assert(forwarded_publication.table_rows == 4);
    assert(forwarded_publication.a_idx_capacity == 3);
    assert(forwarded_buffers.router_computation_num_tokens() == 5);
    const auto& direct_input = forwarded_buffers.nvlink_receive_buffer_for_source(1, 0);
    assert(direct_input.recv.is_alias);
    assert(direct_input.recv.ptr == forwarded_buffers.buffers_for_peer(1).recv.ptr);
    assert(direct_input.expert_token_indices_device.ptr != nullptr);
    assert(forwarded_buffers.nvlink_receive_buffer_for_source(
               1, 1).expert_token_indices_device.ptr != nullptr);
    assert(forwarded_buffers.nvlink_receive_buffer_for_source(
               0, 0).expert_token_indices_device.ptr == nullptr);
    forwarded_buffers.begin_router_notification_iteration(0);
    // Python can enqueue the end event before the first table flush reaches
    // the publication thread; GPU execution still orders the persistent
    // kernel's completion after all readiness-gated table work.
    forwarded_buffers.record_router_computation_end(0, 0);
    forwarded_buffers.process_router_notification_completion(1, 0, 0, 0, 3);
    forwarded_buffers.process_router_notification_completion(1, 1, 0, 0, 3);
    bool rejected_mock_gpu_timing = false;
    try {
        (void)forwarded_buffers.router_computation_elapsed_ms(0);
    } catch (const std::runtime_error&) {
        rejected_mock_gpu_timing = true;
    }
    assert(rejected_mock_gpu_timing);

    DynamicChunkDistributor distributor(
        chunks,
        /*peer_rank=*/2,
        /*qp_count=*/10,
        /*local_base=*/0x1000,
        /*local_lkey=*/11,
        /*remote_base=*/0x2000,
        /*remote_rkey=*/22);
    SendTask task;
    assert(distributor.next(/*qp_index=*/7, task));
    assert(task.chunk.chunk_index == 0);
    assert(task.chunk.qp_index == 7);
    assert(task.signaled);
    assert(task.local_base == 0x1000);
    assert(task.remote_base == 0x2000);
    assert(distributor.next(/*qp_index=*/3, task));
    assert(task.chunk.chunk_index == 1);
    assert(task.chunk.qp_index == 3);
    const auto assignment = distributor.assignment();
    assert(assignment.assigned_chunks() == 2);
    assert(assignment.qp_by_chunk[0] == 7);
    assert(assignment.qp_by_chunk[1] == 3);
    assert(assignment.chunks_by_qp[7] == 1);
    assert(assignment.chunks_by_qp[3] == 1);
    assert(assignment.expected_send_completions_by_qp[7] == 1);
    assert(assignment.expected_send_completions_by_qp[3] == 1);

    std::cout << "test_chunking passed\n";
    return 0;
}
