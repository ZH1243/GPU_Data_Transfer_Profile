#pragma once

#include "config.hpp"
#include "protocol.hpp"
#include "rdma_context.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rdma_proxy {

enum class CompletionKind {
    kSend,
    kRecvWithImmediate,
};

struct Completion {
    CompletionKind kind{CompletionKind::kSend};
    uint64_t wr_id{0};
    uint32_t imm_data{0};
    std::size_t byte_len{0};
};

class RdmaQueuePair {
public:
    RdmaQueuePair(RdmaContext& context, const ProxyConfig& config, int peer_rank, int qp_index);
    ~RdmaQueuePair();

    RdmaQueuePair(const RdmaQueuePair&) = delete;
    RdmaQueuePair& operator=(const RdmaQueuePair&) = delete;

    QPInfo local_info() const;
    void connect(const QPInfo& remote);
    void post_receive(uint64_t wr_id);
    void post_write_with_immediate(
        uint64_t wr_id,
        uintptr_t local_addr,
        uint32_t local_lkey,
        uintptr_t remote_addr,
        uint32_t remote_rkey,
        std::size_t length,
        uint32_t imm_data);
    void post_send_with_immediate(uint64_t wr_id, uint32_t imm_data);
    int poll(std::vector<Completion>& completions, int max_entries);

    int peer_rank() const { return peer_rank_; }
    int qp_index() const { return qp_index_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    RdmaContext& context_;
    ProxyConfig config_;
    int peer_rank_{-1};
    int qp_index_{-1};
};

class ConnectionManager {
public:
    explicit ConnectionManager(ProxyConfig config);

    PeerConnectionInfo exchange_peer_info(
        const PeerAddress& peer,
        const PeerConnectionInfo& local_info) const;
    std::string exchange_control_message(
        const PeerAddress& peer,
        const std::string& local_payload,
        uint64_t timeout_ms) const;

private:
    ProxyConfig config_;
};

}  // namespace rdma_proxy
