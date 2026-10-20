#include "rdma_proxy_c_api.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct ExternalAllocations {
    std::vector<void*> pointers;
    std::vector<RdmaProxyDeviceBufferRequest> requests;
};

uintptr_t allocate_external_buffer(
    void* context,
    const RdmaProxyDeviceBufferRequest* request) {
    if (!context || !request || request->struct_size != sizeof(*request) ||
        request->version != 1 || request->bytes == 0) {
        return 0;
    }
    auto* allocations = static_cast<ExternalAllocations*>(context);
    void* pointer = std::malloc(static_cast<std::size_t>(request->bytes));
    if (!pointer) return 0;
    std::memset(pointer, 0, static_cast<std::size_t>(request->bytes));
    allocations->pointers.push_back(pointer);
    allocations->requests.push_back(*request);
    return reinterpret_cast<uintptr_t>(pointer);
}

}  // namespace

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

    RdmaProxyHandle* handle = rdma_proxy_create(proxy_argc, proxy_argv);
    if (!handle) {
        std::cerr << "per-iteration proxy creation failed: " << rdma_proxy_last_error() << '\n';
        return 1;
    }
    ExternalAllocations external_allocations;
    if (rdma_proxy_set_device_buffer_allocator(
            handle, allocate_external_buffer, &external_allocations) != 0) {
        std::cerr << "failed to install external allocator: "
                  << rdma_proxy_last_error() << '\n';
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_initialize(handle) != 0) {
        std::cerr << "per-iteration proxy initialization failed: " << rdma_proxy_last_error() << '\n';
        rdma_proxy_destroy(handle);
        return 1;
    }
    std::size_t send_request_count = 0;
    std::size_t receive_request_count = 0;
    bool invalid_request = false;
    for (const auto& request : external_allocations.requests) {
        send_request_count +=
            request.kind == RDMA_PROXY_DEVICE_BUFFER_RDMA_SEND ? 1 : 0;
        receive_request_count +=
            request.kind == RDMA_PROXY_DEVICE_BUFFER_RDMA_RECEIVE ? 1 : 0;
        invalid_request = invalid_request || request.dimension_count != 2 ||
            request.dimensions[0] != 32 || request.dimensions[1] != 8 ||
            request.bytes != 32 * 8 * 2;
    }
    if (external_allocations.requests.size() != 6 ||
        send_request_count != 3 || receive_request_count != 3 ||
        invalid_request) {
        std::cerr << "external allocator received unexpected buffer requests\n";
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        for (void* pointer : external_allocations.pointers) std::free(pointer);
        return 1;
    }
    if (rdma_proxy_set_device_buffer_allocator(
            handle, allocate_external_buffer, &external_allocations) == 0) {
        std::cerr << "external allocator was accepted after initialization\n";
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        for (void* pointer : external_allocations.pointers) std::free(pointer);
        return 1;
    }
    uint64_t num_iterations = 0;
    if (rdma_proxy_get_num_iterations(handle, &num_iterations) != 0 || num_iterations != 1) {
        std::cerr << "per-iteration proxy reported unexpected iteration count: "
                  << rdma_proxy_last_error() << '\n';
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_prepare_iteration(handle, 0) != 0) {
        std::cerr << "per-iteration proxy prepare failed: "
                  << rdma_proxy_last_error() << '\n';
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        return 1;
    }
    int iteration_status = -1;
    std::string iteration_error;
    std::thread iteration_thread([&] {
        iteration_status = rdma_proxy_run_iteration(handle, 0);
        if (iteration_status != 0) iteration_error = rdma_proxy_last_error();
    });
    iteration_thread.join();
    if (iteration_status != 0) {
        std::cerr << "per-iteration proxy run failed: " << iteration_error << '\n';
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_finish(handle) != 0) {
        std::cerr << "per-iteration proxy finish failed: " << rdma_proxy_last_error() << '\n';
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_shutdown(handle) != 0) {
        std::cerr << "per-iteration proxy shutdown failed: " << rdma_proxy_last_error() << '\n';
        rdma_proxy_destroy(handle);
        return 1;
    }
    rdma_proxy_destroy(handle);
    // The proxy borrows callback allocations and must leave freeing them to
    // the embedding runtime.
    for (void* pointer : external_allocations.pointers) std::free(pointer);

    // Two local mock proxies are needed for CUDA-IPC metadata/notification setup.
    const auto nonce = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::string exchange = "--nvlink_forward_exchange_dir=/tmp/test_c_api_" + nonce;
    const std::string run_id = "--local_iteration_sync_run_id=test_c_api_" + nonce;
    bool counts_ok[2] = {false, false};
    std::thread query_threads[2];
    for (int gpu = 0; gpu < 2; ++gpu) {
        query_threads[gpu] = std::thread([&, gpu] {
            std::vector<const char*> router_argv(proxy_argv, proxy_argv + proxy_argc);
            const std::string gpu_arg = "--local_gpu_index=" + std::to_string(gpu);
            const std::string device_arg = "--cuda_device_id=" + std::to_string(gpu);
            for (const char* option : {
                     "--num_gpus_per_node=2", "--router_routing_enabled=true",
                     "--nvlink_forwarding_enabled=true", "--num_experts=64",
                     "--nvlink_forward_synchronize_batches=true",
                     "--nvlink_forward_completion_notifications_enabled=true",
                     "--nvlink_forward_threshold_chunks=1", "--nvlink_forward_min_threshold_chunks=1",
                     "--nvlink_forward_max_threshold_chunks=2",
                     "--nvlink_forward_synchronize_iteration=false",
                     exchange.c_str(), run_id.c_str(), gpu_arg.c_str(), device_arg.c_str()}) {
                router_argv.push_back(option);
            }
            auto* router = rdma_proxy_create(static_cast<int>(router_argv.size()), router_argv.data());
            uint64_t n = 0;
            if (!router || rdma_proxy_get_computation_expert_counts(router, nullptr, 0, &n) == 0 ||
                rdma_proxy_initialize(router) != 0) {
                std::cerr << "router query setup failed: " << rdma_proxy_last_error() << '\n';
                rdma_proxy_destroy(router);
                return;
            }
            bool ok = rdma_proxy_get_computation_expert_counts(router, nullptr, 0, &n) == 0 && n == 8;
            std::vector<uint64_t> counts(n);
            uint64_t total = 0;
            ok = ok && rdma_proxy_get_computation_expert_counts(router, counts.data(), n, &n) == 0 &&
                rdma_proxy_get_computation_num_tokens(router, &total) == 0;
            uint64_t sum = 0;
            for (auto count : counts) sum += count;
            counts_ok[gpu] = ok && sum == total && total > 0 &&
                rdma_proxy_get_computation_expert_counts(router, counts.data(), 1, &n) != 0 &&
                rdma_proxy_get_computation_expert_counts(router, nullptr, 1, &n) != 0;
            uintptr_t address = 0;
            int32_t groups[3]{};
            double times[3]{};
            uint64_t observed = 0;
            bool trace_ok = rdma_proxy_configure_down_ready_trace(router, 3, &address) == 0 &&
                address != 0 && rdma_proxy_prepare_iteration(router, 0) == 0;
            if (trace_ok) {
                // Mock mapped storage; the real notification receiver observes
                // this publication, exercising the ABI and receiver integration.
                __atomic_store_n(reinterpret_cast<int32_t*>(address), 3, __ATOMIC_RELEASE);
                trace_ok = rdma_proxy_read_down_ready_trace(
                    router, 0, groups, times, 3, &observed) == 0 &&
                    observed == 1 && groups[0] == 3 && times[0] >= 0 &&
                    rdma_proxy_read_down_ready_trace(router, 0, groups, times, 0, &observed) != 0 &&
                    rdma_proxy_read_down_ready_trace(router, 1, groups, times, 3, &observed) != 0;
            }
            counts_ok[gpu] = counts_ok[gpu] && trace_ok;
            rdma_proxy_shutdown(router);
            rdma_proxy_destroy(router);
        });
    }
    for (auto& thread : query_threads) thread.join();
    if (!counts_ok[0] || !counts_ok[1]) {
        std::cerr << "per-expert C ABI counts or capacity validation failed\n";
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
