#include "rdma_proxy_c_api.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (rdma_proxy_abi_version() != 1) {
        std::cerr << "unexpected RDMA proxy C ABI version\n";
        return 1;
    }
    if (argc != 2) {
        std::cerr << "usage: test_c_api <config>\n";
        return 1;
    }

    const char* proxy_argv[] = {
        "rdma_cpu_proxy",
        "--config",
        argv[1],
        "--cpu_affinity=none",
        "--num_tokens=32",
        "--token_dimension=8",
        "--tokens_per_chunk=8",
        "--num_qps_per_peer=2",
        "--num_iterations=1",
        "--completion_timeout_ms=5000",
        "--fill_test_data=false",
        "--validate_data=false",
        "--log_level=error",
    };
    constexpr int proxy_argc = static_cast<int>(sizeof(proxy_argv) / sizeof(proxy_argv[0]));
    if (rdma_proxy_run_argv(proxy_argc, proxy_argv) != 0) {
        std::cerr << "shared-library proxy run failed: " << rdma_proxy_last_error() << '\n';
        return 1;
    }

    const char* invalid_argv[] = {"rdma_cpu_proxy"};
    if (rdma_proxy_create(1, invalid_argv) != nullptr) {
        std::cerr << "invalid proxy arguments unexpectedly created a handle\n";
        return 1;
    }
    if (std::string(rdma_proxy_last_error()).empty()) {
        std::cerr << "invalid proxy arguments did not report an error\n";
        return 1;
    }
    return 0;
}

