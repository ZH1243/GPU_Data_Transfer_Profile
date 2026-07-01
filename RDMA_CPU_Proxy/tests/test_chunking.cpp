#include "protocol.hpp"
#include "qp_worker.hpp"

#include <cassert>
#include <cstddef>
#include <iostream>

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
