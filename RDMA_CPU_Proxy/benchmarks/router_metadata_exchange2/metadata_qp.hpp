#pragma once
#include "rdma_connection.hpp"
#include <array>
#include <cstring>
#include <stdexcept>
#if RDMA_PROXY_HAVE_VERBS
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#endif

namespace rme2 {
// Minimal RC endpoint: same connection parameters as production, but no
// per-poll allocation or token-payload/chunk machinery in the measured path.
class MetadataQp {
#if RDMA_PROXY_HAVE_VERBS
    struct CqDelete { void operator()(ibv_cq* p) const { ibv_destroy_cq(p); } };
    struct QpDelete { void operator()(ibv_qp* p) const { ibv_destroy_qp(p); } };
    std::unique_ptr<ibv_cq, CqDelete> cq;
    std::unique_ptr<ibv_qp, QpDelete> qp;
#endif
    rdma_proxy::QPInfo info{};
    int port, gid;
public:
    MetadataQp(rdma_proxy::RdmaContext& context, const rdma_proxy::ProxyConfig& config)
        : port(config.rdma_port), gid(config.gid_index) {
#if RDMA_PROXY_HAVE_VERBS
        info.lid = context.local_lid();
        info.psn = uint32_t(17 * (1 + config.node_rank * 8 + config.local_gpu_index));
        context.query_gid(info.gid);
        cq.reset(ibv_create_cq(static_cast<ibv_context*>(context.raw_context()), 64, nullptr, nullptr, 0));
        if (!cq) throw std::runtime_error("metadata ibv_create_cq failed");
        ibv_qp_init_attr init{};
        init.send_cq = init.recv_cq = cq.get(); init.qp_type = IBV_QPT_RC;
        init.cap.max_send_wr = init.cap.max_recv_wr = 32;
        init.cap.max_send_sge = init.cap.max_recv_sge = 1;
        qp.reset(ibv_create_qp(static_cast<ibv_pd*>(context.protection_domain()), &init));
        if (!qp) throw std::runtime_error("metadata ibv_create_qp failed");
        info.qp_num = qp->qp_num;
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT; attr.port_num = port;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        if (ibv_modify_qp(qp.get(), &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            throw std::runtime_error("metadata QP INIT failed");
#else
        (void)context; (void)port; (void)gid;
        throw std::runtime_error("two-node mode requires rebuilding with libibverbs");
#endif
    }
    rdma_proxy::QPInfo local_info() const { return info; }
    void connect(const rdma_proxy::QPInfo& remote) {
#if RDMA_PROXY_HAVE_VERBS
        ibv_qp_attr rtr{};
        rtr.qp_state = IBV_QPS_RTR; rtr.path_mtu = IBV_MTU_4096;
        rtr.dest_qp_num = remote.qp_num; rtr.rq_psn = remote.psn;
        rtr.max_dest_rd_atomic = 1; rtr.min_rnr_timer = 12;
        rtr.ah_attr.is_global = gid >= 0; rtr.ah_attr.dlid = remote.lid;
        rtr.ah_attr.port_num = port;
        if (gid >= 0) {
            std::memcpy(rtr.ah_attr.grh.dgid.raw, remote.gid, 16);
            rtr.ah_attr.grh.sgid_index = gid; rtr.ah_attr.grh.hop_limit = 1;
        }
        if (ibv_modify_qp(qp.get(), &rtr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
            throw std::runtime_error("metadata QP RTR failed");
        ibv_qp_attr rts{};
        rts.qp_state = IBV_QPS_RTS; rts.timeout = 14; rts.retry_cnt = 7; rts.rnr_retry = 7;
        rts.sq_psn = info.psn; rts.max_rd_atomic = 1;
        if (ibv_modify_qp(qp.get(), &rts, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
            throw std::runtime_error("metadata QP RTS failed");
#else
        (void)remote;
#endif
    }
    void post_receive(uint64_t epoch) {
#if RDMA_PROXY_HAVE_VERBS
        ibv_recv_wr wr{}, *bad = nullptr;
        wr.wr_id = epoch;
        if (ibv_post_recv(qp.get(), &wr, &bad)) throw std::runtime_error("metadata post receive failed");
#else
        (void)epoch;
#endif
    }
    void post(uint64_t id, uint64_t from, uint32_t lkey, uint64_t to,
              uint32_t rkey, uint32_t bytes, uint32_t source) {
#if RDMA_PROXY_HAVE_VERBS
        ibv_sge sge{};
        sge.addr = from; sge.lkey = lkey; sge.length = bytes;
        ibv_send_wr wr{}, *bad = nullptr;
        wr.wr_id = id; wr.sg_list = &sge; wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; wr.send_flags = IBV_SEND_SIGNALED;
        wr.imm_data = htonl(source); wr.wr.rdma.remote_addr = to; wr.wr.rdma.rkey = rkey;
        if (ibv_post_send(qp.get(), &wr, &bad)) throw std::runtime_error("metadata post HBM write failed");
#else
        (void)id; (void)from; (void)lkey; (void)to; (void)rkey; (void)bytes; (void)source;
#endif
    }
    void poll(std::vector<rdma_proxy::Completion>& completions) {
#if RDMA_PROXY_HAVE_VERBS
        std::array<ibv_wc, 16> entries{};
        const int count = ibv_poll_cq(cq.get(), int(entries.size()), entries.data());
        if (count < 0) throw std::runtime_error("metadata CQ poll failed");
        for (int i = 0; i < count; ++i) {
            const auto& wc = entries[i];
            if (wc.status != IBV_WC_SUCCESS)
                throw std::runtime_error(std::string("metadata CQ error: ") + ibv_wc_status_str(wc.status) +
                    " vendor=" + std::to_string(wc.vendor_err));
            rdma_proxy::Completion c;
            c.wr_id = wc.wr_id; c.byte_len = wc.byte_len;
            if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM && (wc.wc_flags & IBV_WC_WITH_IMM)) {
                c.kind = rdma_proxy::CompletionKind::kRecvWithImmediate;
                c.imm_data = ntohl(wc.imm_data);
            } else if (wc.opcode != IBV_WC_RDMA_WRITE) {
                throw std::runtime_error("unexpected metadata CQ opcode");
            }
            completions.push_back(c); // caller reserves 16 once during setup
        }
#else
        (void)completions;
#endif
    }
};
}
