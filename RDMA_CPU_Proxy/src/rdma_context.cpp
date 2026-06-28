#include "rdma_context.hpp"

#include "logging.hpp"

#include <cstring>
#include <stdexcept>
#include <unordered_map>

#if RDMA_PROXY_HAVE_VERBS
#include <infiniband/verbs.h>
#endif

namespace rdma_proxy {

struct RdmaContext::Impl {
#if RDMA_PROXY_HAVE_VERBS
    ibv_context* context{nullptr};
    ibv_pd* pd{nullptr};
    ibv_device** device_list{nullptr};
    ibv_port_attr port_attr{};
    std::unordered_map<uint64_t, ibv_mr*> mrs;
#endif
    uint32_t mock_next_key{0x1000};
};

RdmaContext::RdmaContext(ProxyConfig config) : impl_(new Impl), config_(std::move(config)) {}

RdmaContext::~RdmaContext() {
#if RDMA_PROXY_HAVE_VERBS
    for (auto& entry : impl_->mrs) {
        if (ibv_dereg_mr(entry.second)) {
            RDMA_PROXY_LOG_WARN("ibv_dereg_mr failed during cleanup for addr=", entry.first);
        }
    }
    impl_->mrs.clear();
    if (impl_->pd && ibv_dealloc_pd(impl_->pd)) {
        RDMA_PROXY_LOG_WARN("ibv_dealloc_pd failed during cleanup");
    }
    if (impl_->context && ibv_close_device(impl_->context)) {
        RDMA_PROXY_LOG_WARN("ibv_close_device failed during cleanup");
    }
    if (impl_->device_list) {
        ibv_free_device_list(impl_->device_list);
    }
#endif
}

void RdmaContext::initialize() {
    if (config_.mock_mode) {
        RDMA_PROXY_LOG_WARN("RDMA context running in explicit mock_mode; no NIC resources will be opened");
        return;
    }

#if RDMA_PROXY_HAVE_VERBS
    int num_devices = 0;
    impl_->device_list = ibv_get_device_list(&num_devices);
    if (!impl_->device_list || num_devices == 0) {
        throw std::runtime_error("ibv_get_device_list found no RDMA devices");
    }

    ibv_device* selected = nullptr;
    for (int i = 0; i < num_devices; ++i) {
        const char* name = ibv_get_device_name(impl_->device_list[i]);
        if (config_.rdma_device_name.empty() || config_.rdma_device_name == name) {
            selected = impl_->device_list[i];
            break;
        }
    }
    if (!selected) {
        throw std::runtime_error("RDMA device not found: " + config_.rdma_device_name);
    }

    impl_->context = ibv_open_device(selected);
    if (!impl_->context) throw std::runtime_error("ibv_open_device failed");
    impl_->pd = ibv_alloc_pd(impl_->context);
    if (!impl_->pd) throw std::runtime_error("ibv_alloc_pd failed");
    if (ibv_query_port(impl_->context, config_.rdma_port, &impl_->port_attr)) {
        throw std::runtime_error("ibv_query_port failed");
    }
    RDMA_PROXY_LOG_INFO("opened RDMA device ", ibv_get_device_name(selected), " port=", static_cast<int>(config_.rdma_port));
#else
    throw std::runtime_error(
        "libibverbs support was not built. Install rdma-core/libibverbs, rebuild, or set mock_mode=true.");
#endif
}

MemoryRegionInfo RdmaContext::register_memory(void* addr, std::size_t length, const std::string& name) {
    if (config_.mock_mode) {
        const uint32_t key = impl_->mock_next_key++;
        RDMA_PROXY_LOG_WARN("mock MR registration for ", name, " addr=", reinterpret_cast<uintptr_t>(addr),
                            " length=", length);
        return MemoryRegionInfo{reinterpret_cast<uint64_t>(addr), key, key, length};
    }

#if RDMA_PROXY_HAVE_VERBS
    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    ibv_mr* mr = ibv_reg_mr(impl_->pd, addr, length, access);
    if (!mr) {
        throw std::runtime_error(
            "ibv_reg_mr failed for " + name +
            ". For GPU memory this requires GPUDirect RDMA support, compatible NVIDIA driver/NIC firmware, "
            "IOMMU/ACS settings, and a verbs provider that accepts CUDA device pointers.");
    }
    impl_->mrs.emplace(reinterpret_cast<uint64_t>(addr), mr);
    RDMA_PROXY_LOG_INFO("registered MR ", name, " addr=", reinterpret_cast<uintptr_t>(addr),
                        " length=", length, " lkey=", mr->lkey, " rkey=", mr->rkey);
    return MemoryRegionInfo{reinterpret_cast<uint64_t>(addr), mr->lkey, mr->rkey, length};
#else
    (void)addr;
    (void)length;
    (void)name;
    throw std::runtime_error("RDMA MR registration requested but libibverbs support was not built");
#endif
}

void RdmaContext::deregister_memory(const MemoryRegionInfo& info) {
    if (config_.mock_mode) return;
#if RDMA_PROXY_HAVE_VERBS
    auto it = impl_->mrs.find(info.addr);
    if (it == impl_->mrs.end()) return;
    if (ibv_dereg_mr(it->second)) throw std::runtime_error("ibv_dereg_mr failed");
    impl_->mrs.erase(it);
#else
    (void)info;
#endif
}

void* RdmaContext::protection_domain() const {
#if RDMA_PROXY_HAVE_VERBS
    return impl_->pd;
#else
    return nullptr;
#endif
}

void* RdmaContext::raw_context() const {
#if RDMA_PROXY_HAVE_VERBS
    return impl_->context;
#else
    return nullptr;
#endif
}

uint16_t RdmaContext::local_lid() const {
    if (config_.mock_mode) return static_cast<uint16_t>(0x10 + config_.node_rank);
#if RDMA_PROXY_HAVE_VERBS
    return impl_->port_attr.lid;
#else
    return 0;
#endif
}

void RdmaContext::query_gid(uint8_t gid_out[16]) const {
    std::memset(gid_out, 0, 16);
    if (config_.mock_mode || config_.gid_index < 0) return;
#if RDMA_PROXY_HAVE_VERBS
    ibv_gid gid{};
    if (ibv_query_gid(impl_->context, config_.rdma_port, config_.gid_index, &gid)) {
        throw std::runtime_error("ibv_query_gid failed");
    }
    std::memcpy(gid_out, gid.raw, 16);
#endif
}

}  // namespace rdma_proxy
