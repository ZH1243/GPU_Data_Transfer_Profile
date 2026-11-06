#pragma once
#include "metadata_qp.hpp"
#include "two_node_layout.hpp"
#include <memory>

namespace rme2 {
// Control descriptors travel through Gloo during setup only. No TCP metadata
// payload or host-staged send buffer is used by this transport.
struct MetadataRdma {
    std::unique_ptr<rdma_proxy::RdmaContext> context;
    std::unique_ptr<MetadataQp> qp;
    rdma_proxy::MemoryRegionInfo send_mr{}, receive_mr{}, remote{};
    std::vector<rdma_proxy::Completion> completions;
    int node, gpu, capacity;
    bool connected = false;

    MetadataRdma(int n, int g, int cap, const char* device, int port, int gid,
                 void* hbm, void* dram) : node(n), gpu(g), capacity(cap) {
        rdma_proxy::ProxyConfig config;
        config.node_rank = node; config.num_nodes = 2; config.local_gpu_index = gpu;
        config.rdma_device_name = device; config.rdma_port = port; config.gid_index = gid;
        config.send_queue_depth = 32; config.recv_queue_depth = 32; config.cq_depth = 64;
        context = std::make_unique<rdma_proxy::RdmaContext>(config);
        context->initialize();
        const auto bytes = two::words(capacity) * sizeof(int32_t);
        send_mr = context->register_memory(hbm, bytes, "metadata relay HBM");
        receive_mr = context->register_memory(dram, bytes, "metadata receive DRAM");
        qp = std::make_unique<MetadataQp>(*context, config);
        completions.reserve(16);
    }
    std::string descriptor() const {
        rdma_proxy::PeerConnectionInfo info;
        info.node_rank = node; info.gpu_index = gpu; info.recv_buffer = receive_mr;
        info.qps.push_back(qp->local_info());
        return rdma_proxy::serialize_peer_info(info);
    }
    void connect(const std::string& descriptor) {
        if (connected) throw std::runtime_error("RDMA already connected");
        const auto info = rdma_proxy::deserialize_peer_info(descriptor);
        if (info.node_rank != 1 - node || info.gpu_index != gpu || info.qps.size() != 1 ||
            !info.recv_buffer.addr || info.recv_buffer.length != receive_mr.length)
            throw std::runtime_error("RDMA peer topology/arena mismatch");
        remote = info.recv_buffer;
        qp->connect(info.qps.front());
        for (int s = 0; s < two::messages; ++s) qp->post_receive(1);
        connected = true;
    }
    void post(uint64_t iteration, int source, int count) {
        const size_t from = two::record_offset((1 - node) * 8 + source, capacity) * 4;
        const size_t to = two::record_offset(node * 8 + source, capacity) * 4;
        qp->post(two::send_id(iteration, source), send_mr.addr + from,
            send_mr.lkey, remote.addr + to, remote.rkey, uint32_t(two::record_bytes(count)), uint32_t(source));
    }
    void post_x3_x4(uint64_t iteration, int count) {
        const size_t offset = two::x3_x4_offset(capacity) * sizeof(int32_t);
        qp->post(two::send_id(iteration, two::x3_x4_tag), send_mr.addr + offset,
            send_mr.lkey, remote.addr + offset, remote.rkey,
            uint32_t(two::x3_x4_bytes(count)), two::x3_x4_tag);
    }
    // Destroy QP before deregistering its memory; caller destroys transport
    // before freeing either CUDA allocation, including on failure.
    ~MetadataRdma() { qp.reset(); context.reset(); }
};
}
