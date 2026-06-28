#pragma once

#include "config.hpp"
#include "protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rdma_proxy {

class RdmaContext {
public:
    explicit RdmaContext(ProxyConfig config);
    ~RdmaContext();

    RdmaContext(const RdmaContext&) = delete;
    RdmaContext& operator=(const RdmaContext&) = delete;

    void initialize();
    MemoryRegionInfo register_memory(void* addr, std::size_t length, const std::string& name);
    void deregister_memory(const MemoryRegionInfo& info);

    void* protection_domain() const;
    void* raw_context() const;
    uint16_t local_lid() const;
    void query_gid(uint8_t gid_out[16]) const;
    bool is_mock() const { return config_.mock_mode; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ProxyConfig config_;
};

}  // namespace rdma_proxy
