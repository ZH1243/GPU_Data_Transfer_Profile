#include "rdma_connection.hpp"

#include "logging.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <netdb.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <sys/select.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#if RDMA_PROXY_HAVE_VERBS
#include <infiniband/verbs.h>
#endif

namespace rdma_proxy {
namespace {

uint32_t make_psn() {
    std::random_device rd;
    return rd() & 0x00ffffffU;
}

std::string gid_to_string(const uint8_t gid[16]) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i > 0 && i % 2 == 0) out << ':';
        out << std::setw(2) << static_cast<unsigned>(gid[i]);
    }
    return out.str();
}

uint64_t host_to_be64(uint64_t value) {
    uint64_t out = 0;
    auto* dst = reinterpret_cast<uint8_t*>(&out);
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>((value >> ((7 - i) * 8)) & 0xffU);
    }
    return out;
}

uint64_t be64_to_host(uint64_t value) {
    const auto* src = reinterpret_cast<const uint8_t*>(&value);
    uint64_t out = 0;
    for (int i = 0; i < 8; ++i) {
        out = (out << 8) | src[i];
    }
    return out;
}

void send_all(int fd, const void* data, std::size_t bytes) {
    const auto* ptr = static_cast<const char*>(data);
    while (bytes > 0) {
        const ssize_t n = ::send(fd, ptr, bytes, 0);
        if (n <= 0) throw std::runtime_error("socket send failed");
        ptr += n;
        bytes -= static_cast<std::size_t>(n);
    }
}

void recv_all(int fd, void* data, std::size_t bytes) {
    auto* ptr = static_cast<char*>(data);
    while (bytes > 0) {
        const ssize_t n = ::recv(fd, ptr, bytes, MSG_WAITALL);
        if (n <= 0) throw std::runtime_error("socket recv failed");
        ptr += n;
        bytes -= static_cast<std::size_t>(n);
    }
}

std::string exchange_payload_client(const PeerAddress& peer, const std::string& payload, uint64_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    int fd = -1;
    while (fd < 0) {
        addrinfo hints{};
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_family = AF_UNSPEC;
        addrinfo* result = nullptr;
        const auto port = std::to_string(peer.port);
        if (getaddrinfo(peer.host.c_str(), port.c_str(), &hints, &result) != 0) {
            throw std::runtime_error("getaddrinfo failed for peer " + peer.host);
        }

        for (auto* rp = result; rp; rp = rp->ai_next) {
            fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
            if (fd == -1) continue;
            if (::connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;
            ::close(fd);
            fd = -1;
        }
        freeaddrinfo(result);
        if (fd >= 0) break;
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("timed out connecting to peer metadata endpoint");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    const uint64_t len = host_to_be64(payload.size());
    send_all(fd, &len, sizeof(len));
    send_all(fd, payload.data(), payload.size());
    uint64_t remote_len_net = 0;
    recv_all(fd, &remote_len_net, sizeof(remote_len_net));
    const auto remote_len = be64_to_host(remote_len_net);
    std::string remote(remote_len, '\0');
    recv_all(fd, remote.data(), remote.size());
    ::close(fd);
    return remote;
}

std::string exchange_payload_server(uint16_t listen_port, const std::string& payload, uint64_t timeout_ms) {
    const int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_any;
    addr.sin6_port = htons(listen_port);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        throw std::runtime_error("bind failed for metadata listener");
    }
    if (::listen(fd, 1) != 0) {
        ::close(fd);
        throw std::runtime_error("listen failed for metadata listener");
    }

    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(fd, &read_set);
    timeval timeout{};
    timeout.tv_sec = static_cast<time_t>(timeout_ms / 1000);
    timeout.tv_usec = static_cast<suseconds_t>((timeout_ms % 1000) * 1000);
    const int ready = ::select(fd + 1, &read_set, nullptr, nullptr, &timeout);
    if (ready <= 0) {
        ::close(fd);
        if (ready == 0) throw std::runtime_error("timed out waiting for peer metadata endpoint");
        throw std::runtime_error("select failed for metadata listener");
    }

    const int peer_fd = ::accept(fd, nullptr, nullptr);
    ::close(fd);
    if (peer_fd < 0) throw std::runtime_error("accept failed for metadata listener");

    uint64_t remote_len_net = 0;
    recv_all(peer_fd, &remote_len_net, sizeof(remote_len_net));
    const auto remote_len = be64_to_host(remote_len_net);
    std::string remote(remote_len, '\0');
    recv_all(peer_fd, remote.data(), remote.size());
    const uint64_t len = host_to_be64(payload.size());
    send_all(peer_fd, &len, sizeof(len));
    send_all(peer_fd, payload.data(), payload.size());
    ::close(peer_fd);
    return remote;
}

}  // namespace

struct RdmaQueuePair::Impl {
    QPInfo local_info;
    QPInfo remote_info;
    std::mutex mock_mutex;
    std::vector<Completion> mock_completions;
#if RDMA_PROXY_HAVE_VERBS
    ibv_cq* cq{nullptr};
    ibv_qp* qp{nullptr};
#endif
};

RdmaQueuePair::RdmaQueuePair(RdmaContext& context, const ProxyConfig& config, int peer_rank, int qp_index)
    : impl_(new Impl), context_(context), config_(config), peer_rank_(peer_rank), qp_index_(qp_index) {
    impl_->local_info.lid = context_.local_lid();
    impl_->local_info.psn = make_psn();
    context_.query_gid(impl_->local_info.gid);

    if (config_.mock_mode) {
        impl_->local_info.qp_num =
            0x100000U + static_cast<uint32_t>(config_.node_rank * 4096 + peer_rank * 64 + qp_index);
        RDMA_PROXY_LOG_INFO("created mock QP local_rank=", config_.node_rank,
                            " local_gpu=", config_.local_gpu_index,
                            " peer=", peer_rank_,
                            " qp=", qp_index_,
                            " local_qpn=", impl_->local_info.qp_num,
                            " local_lid=", impl_->local_info.lid,
                            " local_psn=", impl_->local_info.psn,
                            " local_gid=", gid_to_string(impl_->local_info.gid));
        return;
    }

#if RDMA_PROXY_HAVE_VERBS
    auto* verbs_context = static_cast<ibv_context*>(context_.raw_context());
    auto* pd = static_cast<ibv_pd*>(context_.protection_domain());
    impl_->cq = ibv_create_cq(verbs_context, config_.cq_depth, nullptr, nullptr, 0);
    if (!impl_->cq) throw std::runtime_error("ibv_create_cq failed");

    ibv_qp_init_attr qp_attr{};
    qp_attr.send_cq = impl_->cq;
    qp_attr.recv_cq = impl_->cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr = static_cast<uint32_t>(config_.send_queue_depth);
    qp_attr.cap.max_recv_wr = static_cast<uint32_t>(config_.recv_queue_depth);
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.cap.max_inline_data = 0;
    impl_->qp = ibv_create_qp(pd, &qp_attr);
    if (!impl_->qp) throw std::runtime_error("ibv_create_qp failed");
    impl_->local_info.qp_num = impl_->qp->qp_num;
    RDMA_PROXY_LOG_INFO("created QP local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " peer=", peer_rank_,
                        " qp=", qp_index_,
                        " local_qpn=", impl_->local_info.qp_num,
                        " local_lid=", impl_->local_info.lid,
                        " local_psn=", impl_->local_info.psn,
                        " local_gid=", gid_to_string(impl_->local_info.gid),
                        " send_wr=", qp_attr.cap.max_send_wr,
                        " recv_wr=", qp_attr.cap.max_recv_wr,
                        " send_sge=", qp_attr.cap.max_send_sge,
                        " recv_sge=", qp_attr.cap.max_recv_sge,
                        " inline=", qp_attr.cap.max_inline_data,
                        " cq_depth=", config_.cq_depth);

    ibv_qp_attr init_attr{};
    init_attr.qp_state = IBV_QPS_INIT;
    init_attr.pkey_index = 0;
    init_attr.port_num = config_.rdma_port;
    init_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    const int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(impl_->qp, &init_attr, flags)) {
        throw std::runtime_error("ibv_modify_qp INIT failed");
    }
#else
    throw std::runtime_error("RdmaQueuePair requested but libibverbs support was not built");
#endif
}

RdmaQueuePair::~RdmaQueuePair() {
#if RDMA_PROXY_HAVE_VERBS
    if (impl_->qp && ibv_destroy_qp(impl_->qp)) {
        RDMA_PROXY_LOG_WARN("ibv_destroy_qp failed during cleanup");
    }
    if (impl_->cq && ibv_destroy_cq(impl_->cq)) {
        RDMA_PROXY_LOG_WARN("ibv_destroy_cq failed during cleanup");
    }
#endif
}

QPInfo RdmaQueuePair::local_info() const {
    return impl_->local_info;
}

QPInfo RdmaQueuePair::remote_info() const {
    return impl_->remote_info;
}

void RdmaQueuePair::connect(const QPInfo& remote) {
    impl_->remote_info = remote;
    if (config_.mock_mode) {
        RDMA_PROXY_LOG_INFO("connected mock QP local_rank=", config_.node_rank,
                            " local_gpu=", config_.local_gpu_index,
                            " peer=", peer_rank_,
                            " qp=", qp_index_,
                            " local_qpn=", impl_->local_info.qp_num,
                            " remote_qpn=", remote.qp_num,
                            " remote_lid=", remote.lid,
                            " remote_psn=", remote.psn,
                            " remote_gid=", gid_to_string(remote.gid));
        return;
    }

#if RDMA_PROXY_HAVE_VERBS
    ibv_qp_attr rtr{};
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = IBV_MTU_4096;
    rtr.dest_qp_num = remote.qp_num;
    rtr.rq_psn = remote.psn;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = config_.gid_index >= 0 ? 1 : 0;
    rtr.ah_attr.dlid = remote.lid;
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = config_.rdma_port;
    if (rtr.ah_attr.is_global) {
        std::memcpy(rtr.ah_attr.grh.dgid.raw, remote.gid, 16);
        rtr.ah_attr.grh.sgid_index = config_.gid_index;
        rtr.ah_attr.grh.hop_limit = 1;
    }
    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(impl_->qp, &rtr, flags)) throw std::runtime_error("ibv_modify_qp RTR failed");

    ibv_qp_attr rts{};
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.sq_psn = impl_->local_info.psn;
    rts.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(impl_->qp, &rts, flags)) throw std::runtime_error("ibv_modify_qp RTS failed");
    RDMA_PROXY_LOG_INFO("connected QP local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " peer=", peer_rank_,
                        " qp=", qp_index_,
                        " local_qpn=", impl_->local_info.qp_num,
                        " remote_qpn=", remote.qp_num,
                        " local_lid=", impl_->local_info.lid,
                        " remote_lid=", remote.lid,
                        " local_psn=", impl_->local_info.psn,
                        " remote_psn=", remote.psn,
                        " local_gid=", gid_to_string(impl_->local_info.gid),
                        " remote_gid=", gid_to_string(remote.gid),
                        " mtu=4096",
                        " sl=", static_cast<int>(rtr.ah_attr.sl),
                        " src_path_bits=", static_cast<int>(rtr.ah_attr.src_path_bits),
                        " gid_index=", config_.gid_index,
                        " rdma_port=", static_cast<int>(config_.rdma_port),
                        " timeout=", static_cast<int>(rts.timeout),
                        " retry_cnt=", static_cast<int>(rts.retry_cnt),
                        " rnr_retry=", static_cast<int>(rts.rnr_retry),
                        " min_rnr_timer=", static_cast<int>(rtr.min_rnr_timer));
#endif
}

void RdmaQueuePair::post_receive(uint64_t wr_id) {
    if (config_.mock_mode) return;
#if RDMA_PROXY_HAVE_VERBS
    ibv_recv_wr wr{};
    wr.wr_id = wr_id;
    wr.sg_list = nullptr;
    wr.num_sge = 0;
    ibv_recv_wr* bad = nullptr;
    if (ibv_post_recv(impl_->qp, &wr, &bad)) throw std::runtime_error("ibv_post_recv failed");
#else
    (void)wr_id;
#endif
}

void RdmaQueuePair::post_write_with_immediate(
    uint64_t wr_id,
    uintptr_t local_addr,
    uint32_t local_lkey,
    uintptr_t remote_addr,
    uint32_t remote_rkey,
    std::size_t length,
    uint32_t imm_data,
    bool signaled) {
    if (config_.mock_mode) {
        std::memcpy(reinterpret_cast<void*>(remote_addr), reinterpret_cast<const void*>(local_addr), length);
        std::lock_guard<std::mutex> lock(impl_->mock_mutex);
        if (signaled) {
            impl_->mock_completions.push_back(Completion{CompletionKind::kSend, wr_id, imm_data, length});
        }
        impl_->mock_completions.push_back(Completion{CompletionKind::kRecvWithImmediate, wr_id, imm_data, length});
        return;
    }

#if RDMA_PROXY_HAVE_VERBS
    ibv_sge sge{};
    if (length > 0) {
        sge.addr = local_addr;
        sge.length = static_cast<uint32_t>(length);
        sge.lkey = local_lkey;
    }

    ibv_send_wr wr{};
    wr.wr_id = wr_id;
    wr.sg_list = length > 0 ? &sge : nullptr;
    wr.num_sge = length > 0 ? 1 : 0;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.imm_data = htonl(imm_data);
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = remote_rkey;

    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(impl_->qp, &wr, &bad)) throw std::runtime_error("ibv_post_send RDMA_WRITE_WITH_IMM failed");
#else
    (void)local_addr;
    (void)local_lkey;
    (void)remote_addr;
    (void)remote_rkey;
    (void)length;
    (void)imm_data;
    throw std::runtime_error("RDMA write requested but libibverbs support was not built");
#endif
}

void RdmaQueuePair::post_send_with_immediate(uint64_t wr_id, uint32_t imm_data) {
    if (config_.mock_mode) {
        std::lock_guard<std::mutex> lock(impl_->mock_mutex);
        impl_->mock_completions.push_back(Completion{CompletionKind::kSend, wr_id, imm_data, 0});
        impl_->mock_completions.push_back(Completion{CompletionKind::kRecvWithImmediate, wr_id, imm_data, 0});
        return;
    }

#if RDMA_PROXY_HAVE_VERBS
    ibv_send_wr wr{};
    wr.wr_id = wr_id;
    wr.sg_list = nullptr;
    wr.num_sge = 0;
    wr.opcode = IBV_WR_SEND_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = htonl(imm_data);

    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(impl_->qp, &wr, &bad)) throw std::runtime_error("ibv_post_send SEND_WITH_IMM failed");
#else
    (void)wr_id;
    (void)imm_data;
    throw std::runtime_error("RDMA send requested but libibverbs support was not built");
#endif
}

int RdmaQueuePair::poll(std::vector<Completion>& completions, int max_entries) {
    if (config_.mock_mode) {
        std::lock_guard<std::mutex> lock(impl_->mock_mutex);
        int n = 0;
        while (n < max_entries && !impl_->mock_completions.empty()) {
            completions.push_back(impl_->mock_completions.front());
            impl_->mock_completions.erase(impl_->mock_completions.begin());
            ++n;
        }
        return n;
    }

#if RDMA_PROXY_HAVE_VERBS
    std::vector<ibv_wc> wc(static_cast<std::size_t>(max_entries));
    const int n = ibv_poll_cq(impl_->cq, max_entries, wc.data());
    if (n < 0) throw std::runtime_error("ibv_poll_cq failed");
    for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            throw std::runtime_error("RDMA completion error status=" + std::to_string(wc[i].status));
        }
        Completion c;
        c.wr_id = wc[i].wr_id;
        c.byte_len = wc[i].byte_len;
        if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM ||
            (wc[i].opcode == IBV_WC_RECV && (wc[i].wc_flags & IBV_WC_WITH_IMM))) {
            c.kind = CompletionKind::kRecvWithImmediate;
            c.imm_data = ntohl(wc[i].imm_data);
        } else {
            c.kind = CompletionKind::kSend;
        }
        completions.push_back(c);
    }
    return n;
#else
    return 0;
#endif
}

ConnectionManager::ConnectionManager(ProxyConfig config) : config_(std::move(config)) {}

PeerConnectionInfo ConnectionManager::exchange_peer_info(
    const PeerAddress& peer,
    const PeerConnectionInfo& local_info) const {
    const auto local_payload = serialize_peer_info(local_info);
    std::string remote_payload;

    if (config_.mock_mode) {
        RDMA_PROXY_LOG_WARN("mock metadata exchange with peer ", peer.node_rank);
        return local_info;
    }

    if (config_.node_rank < peer.node_rank) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        remote_payload = exchange_payload_client(peer, local_payload, config_.completion_timeout_ms);
    } else {
        remote_payload = exchange_payload_server(config_.listen_port, local_payload, config_.completion_timeout_ms);
    }
    auto remote = deserialize_peer_info(remote_payload);
    if (remote.node_rank != peer.node_rank || remote.gpu_index != config_.local_gpu_index) {
        throw std::runtime_error("peer metadata rank/gpu mismatch");
    }
    return remote;
}

std::string ConnectionManager::exchange_control_message(
    const PeerAddress& peer,
    const std::string& local_payload,
    uint64_t timeout_ms) const {
    if (config_.mock_mode) {
        return local_payload;
    }

    if (config_.node_rank < peer.node_rank) {
        return exchange_payload_client(peer, local_payload, timeout_ms);
    }
    return exchange_payload_server(config_.listen_port, local_payload, timeout_ms);
}

}  // namespace rdma_proxy
