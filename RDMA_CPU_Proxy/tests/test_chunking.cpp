#include "protocol.hpp"
#include "qp_worker.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdexcept>
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
