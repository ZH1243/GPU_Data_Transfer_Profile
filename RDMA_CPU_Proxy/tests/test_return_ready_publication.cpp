#include "return_group_progress.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
#if RDMA_PROXY_HAVE_CUDA
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
#endif

void exercise(bool mock) {
    rdma_proxy::ReturnGroupProgress progress;
    progress.configure(3, 1, 3, true, 0, mock);
    const auto addresses = progress.publication_addresses();
    auto* host = reinterpret_cast<int32_t*>(addresses.first);
    auto* device = reinterpret_cast<int32_t*>(addresses.second);
    require(host && device && host != device, "flags must have separate allocations");
#if RDMA_PROXY_HAVE_CUDA
    if (!mock) {
        cudaPointerAttributes attributes{};
        check(cudaPointerGetAttributes(&attributes, host));
        require(attributes.type == cudaMemoryTypeHost, "host flag must be pinned");
        check(cudaPointerGetAttributes(&attributes, device));
        require(attributes.type == cudaMemoryTypeDevice, "device flag must be HBM allocation");
    }
#endif
    auto read_device = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!progress.publication_complete()) {
            require(std::chrono::steady_clock::now() < deadline, "GPU publication timed out");
            std::this_thread::yield();
        }
        int32_t value = -1;
        if (mock) value = __atomic_load_n(device, __ATOMIC_ACQUIRE);
#if RDMA_PROXY_HAVE_CUDA
        else check(cudaMemcpy(&value, device, sizeof(value), cudaMemcpyDeviceToHost));
#endif
        return value;
    };
    for (uint64_t iteration = 0; iteration < 3; ++iteration) {
        progress.reset(iteration);
        require(progress.publication_complete(), "reset must finish before returning");
        require(progress.prefix(iteration) == 0 && read_device() == 0, "stale prefix after reset");
        progress.consume(0, iteration, 0);
        progress.consume(0, iteration, 1);
        progress.consume(0, iteration, 2);
        require(__atomic_load_n(host, __ATOMIC_ACQUIRE) == 0 && read_device() == 0,
                "one peer prematurely advanced publication");
        progress.consume(2, iteration, 0);
        require(progress.prefix(iteration) == 1 && read_device() == 1, "partial prefix missing");
        progress.consume(2, iteration, 1);
        progress.consume(2, iteration, 2);
        require(progress.prefix(iteration) == 3 && read_device() == 3, "final prefix missing");
        require(!progress.consume(0, iteration + 1, 0), "future message consumed early");
        require(read_device() == 3, "future message modified GPU flag");
        require(progress.publication_addresses() == addresses, "addresses changed across iterations");
    }
    progress.shutdown_publication();
    progress.shutdown_publication(); // Idempotent shutdown.
    bool rejected = false;
    try { progress.publication_addresses(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "shutdown exposed freed addresses");
}
}

int main(int argc, char** argv) {
    try {
        const bool use_cuda = argc == 2 && std::string(argv[1]) == "--cuda";
        if (use_cuda) {
#if RDMA_PROXY_HAVE_CUDA
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
                std::cout << "SKIP: no CUDA device\n";
                return 77;
            }
#else
            std::cout << "SKIP: built without CUDA\n";
            return 77;
#endif
        }
        exercise(!use_cuda);
        std::cout << "return-ready publication checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
