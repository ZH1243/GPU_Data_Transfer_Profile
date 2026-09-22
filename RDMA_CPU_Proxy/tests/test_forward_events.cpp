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

        // A completed recording cannot be reused while either consumer owns it.
        using namespace std::chrono_literals;
        rdma_proxy::CudaForwardEventPool pool(2, 0, true);
        auto leases = pool.acquire_batch(2, 100ms);
        auto* original = leases[0].get();
        auto notification = leases[0];
        leases[0]->record(nullptr);
        leases[0]->synchronize();
        leases.clear(); // Also returns the unused batch reservation.
        bool exhausted = false;
        try { (void)pool.acquire_batch(2, 5ms); }
        catch (const std::runtime_error&) { exhausted = true; }
        require(exhausted, "a retained notification lease must prevent recycling");
        notification.reset();
        leases = pool.acquire_batch(2, 100ms);
        require(leases[0].get() == original || leases[1].get() == original,
                "pool must reuse its original event storage");
        for (auto& lease : leases) {
            bool unrecorded = false;
            try { lease->synchronize(); }
            catch (const std::runtime_error&) { unrecorded = true; }
            require(unrecorded, "acquisition must reset the recording state");
            lease->record(nullptr);
            lease->synchronize();
        }
        leases.clear();

        // Returning a completed final lease makes a full reservation possible.
        leases = pool.acquire_batch(2, 100ms);
        for (auto& lease : leases) {
            lease->record(nullptr);
            lease->synchronize();
        }
        std::atomic<bool> acquired{false};
        std::exception_ptr acquire_error;
        std::thread successor([&] {
            try {
                auto next_batch = pool.acquire_batch(2, 5s);
                acquired.store(next_batch.size() == 2);
            } catch (...) { acquire_error = std::current_exception(); }
        });
        leases.clear();
        successor.join();
        if (acquire_error) std::rethrow_exception(acquire_error);
        require(acquired.load(), "released leases must unblock a full batch reservation");

        // The pool storage must outlive a lease even if its owner is destroyed.
        std::shared_ptr<rdma_proxy::CudaForwardEvent> survivor;
        {
            rdma_proxy::CudaForwardEventPool temporary(1, 0, true);
            survivor = temporary.acquire_batch(1, 100ms).front();
        }
        survivor->record(nullptr);
        survivor->synchronize();
        survivor.reset();

        // Release without a successful completion must quarantine the slot.
        rdma_proxy::CudaForwardEventPool uncertain(1, 0, true);
        auto pending = uncertain.acquire_batch(1, 100ms);
        pending[0]->record(nullptr);
        pending.clear();
        exhausted = false;
        try { (void)uncertain.acquire_batch(1, 5ms); }
        catch (const std::runtime_error&) { exhausted = true; }
        require(exhausted, "unconfirmed recording must never be recycled");

        // A blocked reservation must be woken by shutdown, even with live leases.
        leases = pool.acquire_batch(2, 100ms);
        std::atomic<bool> stopped{false};
        std::thread waiter([&] {
            try { (void)pool.acquire_batch(1, 5s); }
            catch (const std::runtime_error&) { stopped.store(true); }
        });
        pool.stop();
        waiter.join();
        require(stopped.load(), "pool shutdown must reject waiting acquisitions");
        leases.clear();

#if RDMA_PROXY_HAVE_CUDA
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "CUDA event checks skipped: no CUDA device\n";
            return 77;
        }
        check(cudaSetDevice(0));
        rdma_proxy::CudaForwardEventPool gpu_pool(2, 0, false);
        auto gpu_leases = gpu_pool.acquire_batch(2, 100ms);
        auto& prefix = *gpu_leases[0];
        auto& end = *gpu_leases[1];
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
        prefix.synchronize();
        require(!end.ready(), "later stream work unexpectedly completed");
        require(cudaStreamQuery(gates.stream) == cudaErrorNotReady,
                "test must leave later work pending after the prefix event completes");
        gates.second.store(true);
        check(cudaStreamSynchronize(gates.stream));
        end.synchronize();
        gpu_leases.clear();
        gpu_leases = gpu_pool.acquire_batch(2, 100ms);
        for (auto& lease : gpu_leases) {
            lease->record(reinterpret_cast<void*>(gates.stream));
            lease->synchronize();
        }
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
