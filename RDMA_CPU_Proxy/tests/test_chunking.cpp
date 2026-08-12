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
    const std::size_t expected_map_entry_bytes = 8 + 32 * 12;
    assert(publication.map_entry_bytes == expected_map_entry_bytes);
    assert(publication.num_map_entries == 3);
    assert(publication.works_per_batch == 32);
    assert(publication.host_map.ptr != nullptr);
    assert(publication.host_map.bytes == 3 * expected_map_entry_bytes);
    assert(publication.host_flag.ptr != nullptr);
    assert(publication.host_flag.bytes == sizeof(uint32_t));
    assert(publication.device_map.ptr != nullptr);
    assert(publication.device_map.bytes == publication.host_map.bytes);
    assert(publication.device_flag.ptr != nullptr);
    assert(publication.device_flag.bytes == sizeof(uint32_t));
    assert(std::all_of(
        static_cast<const uint8_t*>(publication.host_map.ptr),
        static_cast<const uint8_t*>(publication.host_map.ptr) + publication.host_map.bytes,
        [](uint8_t value) { return value == 0; }));
    assert(*static_cast<const uint32_t*>(publication.host_flag.ptr) == 0);
    std::memset(publication.host_map.ptr, 0x5a, publication.host_map.bytes);
    *static_cast<uint32_t*>(publication.host_flag.ptr) = 7;
    std::memset(publication.device_map.ptr, 0xff, publication.device_map.bytes);
    *static_cast<uint32_t*>(publication.device_flag.ptr) = 3;
    router_nvlink_buffers.flush_router_notification_publication();
    assert(std::all_of(
        static_cast<const uint8_t*>(publication.device_map.ptr),
        static_cast<const uint8_t*>(publication.device_map.ptr) + publication.device_map.bytes,
        [](uint8_t value) { return value == 0x5a; }));
    assert(*static_cast<const uint32_t*>(publication.device_flag.ptr) == 7);
    std::memset(publication.host_map.ptr, 0, publication.host_map.bytes);
    *static_cast<uint32_t*>(publication.host_flag.ptr) = 0;
    const auto& installed = router_nvlink_buffers.nvlink_receive_buffer_for_source(2, 3);
    assert(installed.expert_metadata_ready);
    assert(installed.expert_metadata.expert_token_indices ==
           local_expert_metadata.expert_token_indices);
    auto head_state = router_nvlink_buffers.expert_token_head_state_for_source(2, 3);
    assert(!head_state.iteration_initialized);
    assert(head_state.received_token_frontier == 0);
    assert(head_state.expert_token_heads == std::vector<std::size_t>(8, 0));
    assert(head_state.expert_token_tails == std::vector<std::size_t>(8, 0));

    // Flush-only mode publishes map/flag for a completion without touching
    // the source buffer's iteration, frontier, or expert Heads.
    *static_cast<uint32_t*>(publication.host_flag.ptr) = 11;
    *static_cast<uint32_t*>(publication.device_flag.ptr) = 3;
    router_nvlink_buffers.process_router_notification_completion(2, 3, 0, 0, 1);
    head_state = router_nvlink_buffers.expert_token_head_state_for_source(2, 3);
    assert(!head_state.iteration_initialized);
    assert(head_state.received_token_frontier == 0);
    assert(head_state.expert_token_heads == std::vector<std::size_t>(8, 0));
    assert(head_state.expert_token_tails == std::vector<std::size_t>(8, 0));
    assert(*static_cast<const uint32_t*>(publication.device_flag.ptr) == 11);
    *static_cast<uint32_t*>(publication.host_flag.ptr) = 0;

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

    // Build three computation batches for one local expert from two
    // source-specific receive buffers. The final batch has one token, so it
    // must be emitted without waiting for the configured four-token M tile.
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
    assert(scheduler_publication.map_entry_bytes == 32);
    assert(scheduler_publication.num_map_entries == 3);
    assert(scheduler_publication.host_map.bytes == 96);
    assert(scheduler_publication.works_per_batch == 4);
    std::memset(
        scheduler_publication.device_map.ptr, 0xff,
        scheduler_publication.device_map.bytes);

    scheduler_buffers.process_router_notification_completion(0, 0, 0, 0, 2);
    assert(*static_cast<const uint32_t*>(scheduler_publication.device_flag.ptr) == 0);
    assert(std::all_of(
        static_cast<const uint8_t*>(scheduler_publication.device_map.ptr),
        static_cast<const uint8_t*>(scheduler_publication.device_map.ptr) +
            scheduler_publication.device_map.bytes,
        [](uint8_t value) { return value == 0xff; }));

    scheduler_buffers.process_router_notification_completion(0, 1, 0, 0, 3);
    assert(*static_cast<const uint32_t*>(scheduler_publication.device_flag.ptr) == 4);
    const auto* map_bytes = static_cast<const uint8_t*>(
        scheduler_publication.device_map.ptr);
    const auto* row0 = reinterpret_cast<const RouterComputationMapEntryHeader*>(
        map_bytes);
    const auto* row0_info =
        reinterpret_cast<const RouterComputationReceiveBufferInfo*>(
            map_bytes + sizeof(RouterComputationMapEntryHeader));
    assert(row0->global_expert_id == 0);
    assert(row0->num_used_receive_buffers == 2);
    assert(row0_info[0].receive_buffer_id == 0);
    assert(row0_info[0].tail == 0);
    assert(row0_info[0].length == 2);
    assert(row0_info[1].receive_buffer_id == 1);
    assert(row0_info[1].tail == 0);
    assert(row0_info[1].length == 2);
    assert(std::all_of(
        map_bytes + scheduler_publication.map_entry_bytes,
        map_bytes + scheduler_publication.device_map.bytes,
        [](uint8_t value) { return value == 0xff; }));

    scheduler_buffers.process_router_notification_completion(0, 0, 0, 2, 3);
    assert(*static_cast<const uint32_t*>(scheduler_publication.device_flag.ptr) == 4);
    scheduler_buffers.process_router_notification_completion(0, 1, 0, 3, 3);
    assert(*static_cast<const uint32_t*>(scheduler_publication.device_flag.ptr) == 8);
    const auto* row1_bytes = map_bytes + scheduler_publication.map_entry_bytes;
    const auto* row1_info =
        reinterpret_cast<const RouterComputationReceiveBufferInfo*>(
            row1_bytes + sizeof(RouterComputationMapEntryHeader));
    assert(row1_info[0].receive_buffer_id == 0);
    assert(row1_info[0].tail == 2);
    assert(row1_info[0].length == 1);
    assert(row1_info[1].receive_buffer_id == 1);
    assert(row1_info[1].tail == 2);
    assert(row1_info[1].length == 3);

    scheduler_buffers.process_router_notification_completion(0, 1, 0, 6, 1);
    assert(*static_cast<const uint32_t*>(scheduler_publication.device_flag.ptr) == 12);
    const auto* row2_bytes = map_bytes + 2 * scheduler_publication.map_entry_bytes;
    const auto* row2 = reinterpret_cast<const RouterComputationMapEntryHeader*>(
        row2_bytes);
    const auto* row2_info =
        reinterpret_cast<const RouterComputationReceiveBufferInfo*>(
            row2_bytes + sizeof(RouterComputationMapEntryHeader));
    assert(row2->num_used_receive_buffers == 1);
    assert(row2_info[0].receive_buffer_id == 1);
    assert(row2_info[0].tail == 5);
    assert(row2_info[0].length == 1);

    const auto scheduler_state =
        scheduler_buffers.router_computation_scheduler_state();
    assert(scheduler_state.initialized);
    assert(scheduler_state.published_batches == 3);
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

    // A new iteration resets the GPU-visible work frontier even when its
    // first completion has too few tokens to produce a map row.
    scheduler_buffers.process_router_notification_completion(0, 0, 1, 0, 1);
    assert(*static_cast<const uint32_t*>(scheduler_publication.device_flag.ptr) == 0);
    const auto reset_scheduler_state =
        scheduler_buffers.router_computation_scheduler_state();
    assert(reset_scheduler_state.iteration == 1);
    assert(reset_scheduler_state.published_batches == 0);
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
    // the default policy coalesces three appended rows into one publication.
    // Per-entry mode publishes those same three rows independently.
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
        policy_buffers.process_router_notification_completion(0, 0, 0, 0, 7);
        assert(policy_publication.map_flush_count == 0);
        assert(policy_publication.flag_publication_count == 0);
        policy_buffers.process_router_notification_completion(0, 1, 0, 0, 7);
        assert(*static_cast<const uint32_t*>(policy_publication.device_flag.ptr) == 12);
        const auto policy_state =
            policy_buffers.router_computation_scheduler_state();
        assert(policy_state.published_batches == 3);
        return std::make_pair(
            policy_publication.map_flush_count,
            policy_publication.flag_publication_count);
    };
    assert(run_flush_policy(false) ==
           std::make_pair(uint64_t{1}, uint64_t{1}));
    assert(run_flush_policy(true) ==
           std::make_pair(uint64_t{3}, uint64_t{3}));

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
