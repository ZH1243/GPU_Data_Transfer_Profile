#include "cuda_buffers.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
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

// Keep two explicit stream positions pending without relying on GPU speed.
// These callbacks never call CUDA and are released even if an assertion fails.
struct StreamGates {
    cudaStream_t stream{nullptr};
    std::atomic<bool> first{false};
    std::atomic<bool> second{false};
    ~StreamGates() {
        first.store(true);
        second.store(true);
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
};

void CUDART_CB wait_at_gate(void* pointer) {
    auto& gate = *static_cast<std::atomic<bool>*>(pointer);
    while (!gate.load()) std::this_thread::yield();
}
#endif
}  // namespace

int main() {
    try {
        rdma_proxy::CudaForwardEvent mock(true);
        bool rejected = false;
        try { (void)mock.ready(); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "an unrecorded event must not be published as ready");
        mock.record(nullptr);
        require(mock.ready(), "mock copy completion is synchronous");
        rejected = false;
        try { mock.record(nullptr); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "a shared event must not overwrite its captured prefix");

#if RDMA_PROXY_HAVE_CUDA
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "CUDA event checks skipped: no CUDA device\n";
            return 77;
        }
        check(cudaSetDevice(0));
        rdma_proxy::CudaForwardEvent prefix(false);
        rdma_proxy::CudaForwardEvent end(false);
        StreamGates gates;
        check(cudaStreamCreateWithFlags(&gates.stream, cudaStreamNonBlocking));
        check(cudaLaunchHostFunc(gates.stream, wait_at_gate, &gates.first));
        prefix.record(reinterpret_cast<void*>(gates.stream));
        check(cudaLaunchHostFunc(gates.stream, wait_at_gate, &gates.second));
        end.record(reinterpret_cast<void*>(gates.stream));
        require(!prefix.ready(), "event became ready before its stream prefix completed");

        // Handoff and dispatcher consumers may query from different CPU threads.
        std::atomic<bool> observed_not_ready{false};
        std::exception_ptr observer_error;
        std::thread observer([&] {
            try {
                check(cudaSetDevice(0));
                observed_not_ready.store(!prefix.ready());
            } catch (...) { observer_error = std::current_exception(); }
        });
        observer.join();
        if (observer_error) std::rethrow_exception(observer_error);
        require(observed_not_ready.load(), "second CPU consumer lost the event dependency");

        gates.first.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!prefix.ready()) {
            require(std::chrono::steady_clock::now() < deadline, "prefix event timed out");
            std::this_thread::yield();
        }
        require(!end.ready(), "later stream work unexpectedly completed");
        require(cudaStreamQuery(gates.stream) == cudaErrorNotReady,
                "test must leave later work pending after the prefix event completes");
        gates.second.store(true);
        check(cudaStreamSynchronize(gates.stream));
        require(end.ready(), "batch-end event did not complete");
#else
        std::cout << "Mock event checks passed; CUDA prefix checks require a CUDA build\n";
#endif
        std::cout << "test_forward_events passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
