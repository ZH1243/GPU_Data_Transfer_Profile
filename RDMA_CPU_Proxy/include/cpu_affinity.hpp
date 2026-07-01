#pragma once

#include "config.hpp"

namespace rdma_proxy {

void apply_cpu_affinity(const ProxyConfig& config);

}  // namespace rdma_proxy
