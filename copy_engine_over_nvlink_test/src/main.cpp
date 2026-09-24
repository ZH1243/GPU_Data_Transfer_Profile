#include "common.hpp"
#include <cuda_runtime_api.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <set>
#include <thread>
#include <type_traits>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#define CUDA(call) do { auto e = (call); if (e != cudaSuccess) throw std::runtime_error( \
    std::string(#call) + ": " + cudaGetErrorString(e)); } while (false)
struct Config {
    std::vector<int> devices{0,1,2,3,4,5,6,7}, cpus;
    std::vector<size_t> sizes{16000000,12000000,8000000,8000000,6000000,4000000,2000000};
    size_t batches = 5, iterations = 10;
    unsigned timeout = 120;
    bool verify = true;
};
std::vector<int> integers(const std::string& s) {
    std::vector<int> out;
    for (const auto& t : split(s)) {
        auto x = number(t);
        if (x > static_cast<size_t>(std::numeric_limits<int>::max())) throw std::runtime_error("integer too large");
        out.push_back(static_cast<int>(x));
    }
    if (std::set<int>(out.begin(), out.end()).size() != out.size()) throw std::runtime_error("duplicate list entries");
    return out;
}
Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "Usage: nvlink_all_to_all [options]\n"
                "  --devices 0,1,2,3,4,5,6,7  CUDA-visible device ordinals in rank order\n"
                "  --sizes 16MB,12MB,8MB,8MB,6MB,4MB,2MB  Exactly n-1 positive sizes\n"
                "  --batches 5 --iterations 10 --timeout-seconds 120\n"
                "  --cpus 0,1,...  Optional distinct CPU IDs, one per proxy\n"
                "  --no-verify    Skip final full receive-buffer verification\n"
                "Units: B, KB, MB, GB are decimal; KiB, MiB, GiB are binary.\n";
            std::exit(0);
        }
        if (arg == "--no-verify") { c.verify = false; continue; }
        if (++i == argc) throw std::runtime_error("missing value for " + arg);
        std::string value = argv[i];
        if (arg == "--devices") c.devices = integers(value);
        else if (arg == "--cpus") c.cpus = integers(value);
        else if (arg == "--sizes") {
            c.sizes.clear();
            for (auto& s : split(value)) c.sizes.push_back(bytes(s));
        } else if (arg == "--batches") c.batches = number(value);
        else if (arg == "--iterations") c.iterations = number(value);
        else if (arg == "--timeout-seconds") {
            auto t = number(value);
            if (!t || t > 86400) throw std::runtime_error("timeout must be 1..86400 seconds");
            c.timeout = static_cast<unsigned>(t);
        } else throw std::runtime_error("unknown option " + arg);
    }
    auto n = c.devices.size();
    if (n < 2 || n > max_ranks || c.sizes.size() != n - 1)
        throw std::runtime_error("require 2..32 devices and exactly n-1 sizes");
    if (!c.batches || !c.iterations) throw std::runtime_error("batches and iterations must be positive");
    if (!c.cpus.empty() && c.cpus.size() != n) throw std::runtime_error("need one CPU per rank");
    for (auto cpu : c.cpus) if (cpu >= CPU_SETSIZE) throw std::runtime_error("CPU exceeds CPU_SETSIZE");
    checked_mul(c.iterations, checked_add(c.batches, 1));
    return c;
}
struct Shared {
    BarrierState barrier;
    cudaIpcMemHandle_t handles[max_ranks][max_ranks]{}; // [receiver][sender]
};
// CUDA releases expose signatures with and without failIdx; select at compile time.
template<class Fn>
cudaError_t copy_one(Fn fn, void* dst, void* src, size_t size, cudaStream_t stream) {
    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t index = 0;
    if constexpr (std::is_invocable_r_v<cudaError_t, Fn, void**, void**, size_t*, size_t,
                  cudaMemcpyAttributes*, size_t*, size_t, size_t*, cudaStream_t>) {
        size_t fail = SIZE_MAX;
        return fn(&dst, &src, &size, 1, &attrs, &index, 1, &fail, stream);
    } else if constexpr (std::is_invocable_r_v<cudaError_t, Fn, void**, void**, size_t*, size_t,
                         cudaMemcpyAttributes*, size_t*, size_t, cudaStream_t>) {
        return fn(&dst, &src, &size, 1, &attrs, &index, 1, stream);
    } else {
        const void* const_dst = dst;
        const void* const_src = src;
        return fn(&const_dst, &const_src, &size, 1, &attrs, &index, 1, stream);
    }
}
unsigned char pattern(int sender, size_t batch, size_t step) {
    return static_cast<unsigned char>(1 + (sender * 37 + (batch % 251) * 17 + step * 11) % 251);
}
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b-a).count(); }
void worker(const Config& c, const Layout& l, Shared& s, int rank) {
    int n = static_cast<int>(c.devices.size());
    Barrier barrier(s.barrier, rank, n, c.timeout);
    if (!c.cpus.empty()) {
        cpu_set_t mask;
        CPU_ZERO(&mask); CPU_SET(c.cpus[rank], &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask)) throw std::runtime_error("sched_setaffinity failed");
    }
    int count = 0;
    CUDA(cudaGetDeviceCount(&count));
    for (int d : c.devices) if (d >= count) throw std::runtime_error("device ordinal is not visible");
    CUDA(cudaSetDevice(c.devices[rank]));
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, c.devices[rank]));
    if (!prop.unifiedAddressing || prop.computeMode != cudaComputeModeDefault)
        throw std::runtime_error("require UVA and default compute mode for CUDA IPC");
    for (int peer = 0; peer < n; ++peer) if (peer != rank) {
        int can = 0;
        CUDA(cudaDeviceCanAccessPeer(&can, c.devices[rank], c.devices[peer]));
        if (!can) throw std::runtime_error("P2P unavailable to rank " + std::to_string(peer));
    }
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    void* source = nullptr;
    std::vector<void*> receive(n, nullptr), remote(n, nullptr);
    CUDA(cudaMalloc(&source, l.source_bytes));
    for (int sender = 0; sender < n; ++sender) if (sender != rank) {
        int step = (rank - sender + n) % n;
        CUDA(cudaMalloc(&receive[sender], l.receive(step)));
        CUDA(cudaMemsetAsync(receive[sender], 0, l.receive(step), stream));
        CUDA(cudaIpcGetMemHandle(&s.handles[rank][sender], receive[sender]));
    }
    for (size_t b = 0; b < c.batches; ++b)
        for (int step = 1; step < n; ++step)
            CUDA(cudaMemsetAsync(static_cast<char*>(source) + l.src(b, step),
                 pattern(rank, b, step), l.sizes[step-1], stream));
    CUDA(cudaStreamSynchronize(stream));
    barrier.wait(); // Publish handles and finish all allocation/initialization.
    for (int peer = 0; peer < n; ++peer) if (peer != rank)
        CUDA(cudaIpcOpenMemHandle(&remote[peer], s.handles[peer][rank], cudaIpcMemLazyEnablePeerAccess));
    barrier.wait();
    struct Result { double copy = 0, wall = 0; };
    std::vector<Result> results(c.iterations);
    for (size_t iteration = 0; iteration < c.iterations; ++iteration) {
        barrier.wait(); // Beginning of every iteration; offsets reset below.
        auto start = Clock::now();
        for (size_t b = 0; b < c.batches; ++b) {
            auto begin = Clock::now();
            for (int step = 1; step < n; ++step) {
                int peer = (rank + step) % n;
                CUDA(copy_one(&cudaMemcpyBatchAsync,
                    static_cast<char*>(remote[peer]) + l.dst(b, step),
                    static_cast<char*>(source) + l.src(b, step), l.sizes[step-1], stream));
            }
            CUDA(cudaStreamSynchronize(stream));
            auto done = Clock::now();
            barrier.wait(); // Every sender has completed, including after last batch.
            results[iteration].copy += seconds(begin, done);
        }
        results[iteration].wall = seconds(start, Clock::now());
    }
    if (c.verify) {
        std::vector<unsigned char> host(std::min<size_t>(l.source_bytes, 4 * 1024 * 1024));
        for (int sender = 0; sender < n; ++sender) if (sender != rank) {
            int step = (rank - sender + n) % n;
            for (size_t b = 0; b < c.batches; ++b) {
                for (size_t off = 0; off < l.sizes[step-1];) {
                    size_t len = std::min(host.size(), l.sizes[step-1] - off);
                    CUDA(cudaMemcpy(host.data(), static_cast<char*>(receive[sender]) + l.dst(b, step) + off,
                                    len, cudaMemcpyDeviceToHost));
                    if (!std::all_of(host.begin(), host.begin() + len,
                        [&](unsigned char v) { return v == pattern(sender, b, step); }))
                        throw std::runtime_error("verification failed: sender=" + std::to_string(sender) + " batch=" + std::to_string(b));
                    off += len;
                }
            }
        }
    }
    barrier.wait();
    // No exporter frees an allocation until every importer has closed its mapping.
    for (auto p : remote) if (p) CUDA(cudaIpcCloseMemHandle(p));
    barrier.wait();
    for (auto p : receive) if (p) CUDA(cudaFree(p));
    CUDA(cudaFree(source));
    CUDA(cudaStreamDestroy(stream));
    for (int turn = 0; turn < n; ++turn) {
        if (rank == turn) {
            std::printf("# rank=%d device=%d name=%s verification=%s\n", rank, c.devices[rank], prop.name, c.verify ? "PASS" : "disabled");
            for (size_t it = 0; it < c.iterations; ++it)
                std::printf("%d,%zu,%zu,%.6f,%.6f,%.6f\n", rank, it, l.source_bytes,
                    results[it].copy * 1000, results[it].wall * 1000,
                    static_cast<double>(l.source_bytes) / results[it].wall / 1e9);
            std::fflush(stdout);
        }
        barrier.wait();
    }
}
volatile sig_atomic_t interrupted = 0;
void signal_handler(int) { interrupted = 1; }
int main(int argc, char** argv) {
    try {
        auto c = parse(argc, argv);
        Layout l(c.sizes, c.batches);
        void* mem = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (mem == MAP_FAILED) throw std::runtime_error("mmap failed");
        auto* s = new(mem) Shared{};
        std::printf("# ranks=%zu batches=%zu iterations=%zu src_bytes_per_gpu=%zu recv_bytes_per_gpu=%zu\n",
            c.devices.size(), c.batches, c.iterations, l.source_bytes, l.source_bytes);
        std::printf("rank,iteration,sent_bytes,submit_and_sync_ms,iteration_with_barriers_ms,sent_GBps\n");
        std::fflush(stdout);
        std::signal(SIGINT, signal_handler); std::signal(SIGTERM, signal_handler);
        std::vector<pid_t> children;
        children.reserve(c.devices.size());
        pid_t parent = getpid();
        bool failed = false;
        // Parent never initializes CUDA: each child creates its own CUDA context.
        for (size_t rank = 0; rank < c.devices.size(); ++rank) {
            pid_t pid = fork();
            if (pid < 0) { failed = true; break; }
            if (pid == 0) {
                std::signal(SIGINT, SIG_DFL); std::signal(SIGTERM, SIG_DFL);
                if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
                try { worker(c, l, *s, static_cast<int>(rank)); _exit(0); }
                catch (const std::exception& e) {
                    std::fprintf(stderr, "rank %zu: %s\n", rank, e.what());
                    abort_all(s->barrier); _exit(1);
                }
            }
            children.push_back(pid);
        }
        size_t remaining = children.size();
        bool killed = false;
        while (remaining) {
            if (failed || interrupted || aborted(s->barrier)) {
                failed = true;
                if (!killed) {
                    abort_all(s->barrier);
                    for (pid_t p : children) if (p > 0) kill(p, SIGKILL);
                    killed = true;
                }
            }
            for (auto& p : children) if (p > 0) {
                int status = 0;
                pid_t done = waitpid(p, &status, WNOHANG);
                if (done == p) {
                    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) failed = true;
                    p = -1; --remaining;
                } else if (done < 0 && errno != EINTR) { failed = true; p = -1; --remaining; }
            }
            if (remaining) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        s->~Shared(); munmap(mem, sizeof(Shared));
        if (failed || interrupted) { std::fprintf(stderr, "benchmark failed\n"); return 1; }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
