#include "proxy.hpp"
#include "logging.hpp"
#include <iostream>
#include <stdexcept>

namespace rdma_proxy {
void require_dual(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct DualRdmaTestAccess {
    static void exercise() {
        ProxyConfig c;
        c.mock_mode = true;
        c.num_nodes = 2;
        c.num_gpus_per_node = 1;
        c.num_tokens = 65;
        c.token_dimension = 8;
        c.tokens_per_chunk = 7;
        c.num_qps_per_peer = 20;
        c.num_iterations = 3;
        c.fill_test_data = c.validate_data = true;
        c.rdma_device_name = "mock0";
        c.rdma_device_name2 = "mock1";
        c.rdma_port2 = 2;
        c.gid_index2 = 7;
        c.peers = {PeerAddress{1, "mock", 12345}};
        validate_config(c);
        for (int invalid : {1, 3, 19}) {
            auto bad = c; bad.num_qps_per_peer = invalid;
            bool rejected = false;
            try { validate_config(bad); } catch (const std::runtime_error&) { rejected = true; }
            require_dual(rejected, "odd dual-device QP count accepted");
        }
        {
            auto bad = c; bad.rdma_device_name2 = bad.rdma_device_name;
            bool rejected = false;
            try { validate_config(bad); } catch (const std::runtime_error&) { rejected = true; }
            require_dual(rejected, "duplicate device accepted");
        }
        Proxy proxy(c);
        proxy.initialize();
        auto& peer = proxy.peers_.front();
        require_dual(peer.workers.size() == 20, "QP count changed");
        for (int q = 0; q < 20; ++q) {
            require_dual(&proxy.qp_context(q) == (q < 10 ? &proxy.rdma_context_ : proxy.rdma_context2_.get()),
                         "wrong QP context");
            require_dual(proxy.qp_config(q).rdma_device_name == (q < 10 ? "mock0" : "mock1"),
                         "wrong QP device");
            if (q >= 10) require_dual(proxy.qp_config(q).rdma_port == 2 &&
                                      proxy.qp_config(q).gid_index == 7, "device2 port/GID lost");
        }
        auto info = proxy.make_local_peer_info(peer);
        // Force distinct keys: mock PDs otherwise start their key counters identically.
        info.recv_buffer.rkey = 101; info.recv_buffer2.rkey = 202;
        const auto wire = serialize_peer_info(info);
        const auto parsed = deserialize_peer_info(wire);
        require_dual(parsed.recv_buffer.rkey == 101 && parsed.recv_buffer2.rkey == 202 &&
                     parsed.qps.size() == 20, "dual metadata roundtrip lost keys");
        bool rejected = false;
        try { deserialize_peer_info(wire.substr(0, wire.rfind(' '))); }
        catch (const std::runtime_error&) { rejected = true; }
        require_dual(rejected, "truncated dual metadata accepted");
        auto legacy = info; legacy.recv_buffer2 = {};
        require_dual(deserialize_peer_info(serialize_peer_info(legacy)).recv_buffer2.addr == 0,
                     "single-device metadata changed");

        auto chunks = compute_chunks(23, 8, 2, 1, 20);
        DynamicChunkDistributor distributor(chunks, 1, 20, 0, 0, 0, 0);
        MemoryRegionInfo a{1000, 11, 12, 4096}, b{1000, 21, 22, 4096};
        MemoryRegionInfo ra{2000, 31, 32, 4096}, rb{2000, 41, 42, 4096};
        proxy.configure_distributor_memory(distributor, 20, a, ra, b, rb);
        for (int i = 0; i < 23; ++i) {
            const int q = (i * 7) % 20;
            SendTask task;
            require_dual(distributor.next(q, task), "chunk lost");
            require_dual(task.chunk.chunk_index == static_cast<std::size_t>(i) &&
                         task.local_lkey == (q < 10 ? 11U : 21U) &&
                         task.remote_rkey == (q < 10 ? 32U : 42U) &&
                         task.local_base == 1000 && task.remote_base == 2000,
                         "incorrect per-device send addressing");
        }
        SendTask task;
        require_dual(!distributor.next(0, task), "chunk duplicated");
        proxy.run(); // Actual mock payload copies, tails, markers, and repeated iterations.
        proxy.shutdown();
    }
};
}
int main() {
    rdma_proxy::Logger::instance().set_level(rdma_proxy::LogLevel::kError);
    rdma_proxy::DualRdmaTestAccess::exercise();
    std::cout << "dual-device mock tests passed\n";
}
