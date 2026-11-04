#include "native_proxy.h"
#include "exchange.cuh"
#include "metadata_view.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
thread_local std::string last_error;
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
template<class F> int api(F&& f) {
    try { last_error.clear(); f(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
    catch (...) { last_error = "unknown native exception"; return -1; }
}
// Linux process-shared, lock-free aligned atomic counters. Only CPU threads
// touch these; GPU readiness comes from a completed CUDA event first.
struct alignas(64) Slot { uint64_t complete; char padding[56]; };
struct Shared { Slot slots[rme2::gpus]; Slot aborted; };
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr), "64-bit atomics required");
uint64_t acquire(const uint64_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
void release(uint64_t* p, uint64_t value) { __atomic_store_n(p, value, __ATOMIC_RELEASE); }
}

struct Rme2Proxy {
    int rank, device, tokens, capacity, core;
    double timeout;
    bool validate, owner = false, connected = false, pending = false, done = false;
    std::string name, failure;
    Shared* shared = nullptr;
    int32_t* inbox = nullptr;
    int32_t* pinned = nullptr;
    PeerArenas peers{};
    std::array<bool, rme2::gpus> opened{};
    cudaStream_t read_stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr, read_done = nullptr;
    std::array<rme2::Installed, rme2::gpus> installed{};
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<bool> stopping{false};
    uint64_t sequence = 0;
    const int32_t* local_values = nullptr;
    Clock::time_point submitted;
    Rme2Stats stats{};

    Rme2Proxy(int r, int d, int t, const char* n, double seconds, bool v, int c)
        : rank(r), device(d), tokens(t), capacity(t * 8), core(c), timeout(seconds),
          validate(v), name(n) {}

    size_t words() const { return size_t(rme2::gpus) * (rme2::header_words + size_t(capacity)); }
    int32_t* payload(int32_t* arena, int source) const {
        return arena + rme2::gpus * rme2::header_words + size_t(source) * capacity;
    }
    void initialize() {
        check(cudaSetDevice(device), "select GPU");
        // Dedicated allocations: IPC does not expose a PyTorch allocator slab.
        check(cudaMalloc(reinterpret_cast<void**>(&inbox), words() * 4), "allocate IPC inbox");
        check(cudaMallocHost(reinterpret_cast<void**>(&pinned), words() * 4), "allocate pinned records");
        std::memset(pinned, 0, words() * 4); // prefault before timing
        check(cudaMemset(inbox, 0, words() * 4), "initialize IPC inbox");
        check(cudaStreamCreateWithFlags(&read_stream, cudaStreamNonBlocking), "create readback stream");
        check(cudaEventCreate(&begin), "create exchange start event");
        check(cudaEventCreate(&end), "create exchange end event");
        check(cudaEventCreateWithFlags(&read_done, cudaEventDisableTiming), "create readback event");
        int fd = shm_open(name.c_str(), O_RDWR | (rank == 0 ? O_CREAT | O_EXCL : 0), 0600);
        require(fd >= 0, "shm_open failed (rank 0 must create first; check /dev/shm)");
        owner = rank == 0;
        if (owner && ftruncate(fd, sizeof(Shared)) != 0) {
            close(fd); throw std::runtime_error("ftruncate shared control failed");
        }
        struct stat status{};
        if (fstat(fd, &status) != 0 || status.st_size != sizeof(Shared)) {
            close(fd); throw std::runtime_error("invalid shared control size");
        }
        void* mapping = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        require(mapping != MAP_FAILED, "mmap shared control failed");
        shared = static_cast<Shared*>(mapping);
        if (owner) std::memset(shared, 0, sizeof(Shared));
        check(cudaDeviceSynchronize(), "finish inbox initialization");
    }
    void check_deadline(Clock::time_point start) {
        if (stopping.load() || acquire(&shared->aborted.complete))
            throw std::runtime_error("metadata exchange aborted by a peer or shutdown");
        if (ms(start) > timeout * 1000)
            throw std::runtime_error("metadata exchange timeout (CUDA completion or missing peer)");
    }
    void wait_event(cudaEvent_t event) {
        auto start = Clock::now();
        for (;;) {
            auto status = cudaEventQuery(event);
            if (status == cudaSuccess) return;
            if (status != cudaErrorNotReady) check(status, "poll CUDA completion");
            check_deadline(start);
            std::this_thread::yield();
        }
    }
    void process() {
        stats = {};
        auto start = Clock::now();
        wait_event(end);
        stats.sender_completion_wait_ms = ms(start);
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, begin, end), "exchange event timing");
        stats.exchange_gpu_ms = elapsed;
        // Completion covers every CTA's system-fenced peer writes. Publication
        // occurs only after the source event completes, never at kernel launch.
        release(&shared->slots[rank].complete, sequence);
        start = Clock::now();
        for (int source = 0; source < rme2::gpus; ++source) {
            while (acquire(&shared->slots[source].complete) < sequence) {
                check_deadline(start);
                std::this_thread::yield();
            }
        }
        stats.peer_completion_wait_ms = ms(start);
        start = Clock::now();
        stats.header_bytes = rme2::gpus * rme2::header_words * sizeof(int32_t);
        check(cudaMemcpyAsync(pinned, inbox, stats.header_bytes, cudaMemcpyDeviceToHost,
                              read_stream), "copy headers to pinned records");
        check(cudaEventRecord(read_done, read_stream), "record header readback");
        wait_event(read_done);
        stats.headers_d2h_ms = ms(start);
        start = Clock::now();
        for (int source = 0; source < rme2::gpus; ++source) {
            const int32_t* header = pinned + source * rme2::header_words;
            int count = rme2::check_header(header, capacity);
            auto bytes = size_t(count) * sizeof(int32_t);
            const int32_t* src = source == rank ? local_values + header[0] : payload(inbox, source);
            if (bytes) check(cudaMemcpyAsync(payload(pinned, source), src, bytes,
                                            cudaMemcpyDeviceToHost, read_stream), "copy exact metadata payload");
            stats.received_index_bytes += bytes;
            if (source != rank) stats.peer_received_index_bytes += bytes;
        }
        check(cudaEventRecord(read_done, read_stream), "record payload readback");
        wait_event(read_done);
        stats.payload_d2h_ms = ms(start);
        start = Clock::now();
        for (int source = 0; source < rme2::gpus; ++source)
            rme2::install(installed[source], pinned + source * rme2::header_words,
                          payload(pinned, source), tokens, sequence, validate);
        stats.install_ms = ms(start);
        stats.native_ms = ms(submitted);
    }
    void run() noexcept {
        try {
            check(cudaSetDevice(device), "select proxy thread GPU");
            if (core >= 0) {
                cpu_set_t cpus;
                CPU_ZERO(&cpus); CPU_SET(core, &cpus);
                require(pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) == 0,
                        "proxy CPU affinity failed");
            }
            for (;;) {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return stopping.load() || pending; });
                    if (stopping.load()) return;
                }
                process();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    pending = false; done = true;
                }
                changed.notify_all();
            }
        } catch (const std::exception& e) {
            if (shared) release(&shared->aborted.complete, 1);
            {
                std::lock_guard<std::mutex> lock(mutex);
                failure = e.what(); pending = false; done = true;
            }
            changed.notify_all();
        }
    }
    ~Rme2Proxy() {
        stopping.store(true);
        changed.notify_all();
        if (worker.joinable()) worker.join();
        cudaSetDevice(device);
        // Caller must collectively finish before destroying IPC allocations.
        if (read_stream) cudaStreamSynchronize(read_stream);
        if (end) cudaEventSynchronize(end);
        for (int p = 0; p < rme2::gpus; ++p)
            if (opened[p]) cudaIpcCloseMemHandle(peers.ptr[p]);
        if (read_stream) cudaStreamDestroy(read_stream);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (read_done) cudaEventDestroy(read_done);
        if (inbox) cudaFree(inbox);
        if (pinned) cudaFreeHost(pinned);
        if (shared) munmap(shared, sizeof(Shared));
        if (owner) shm_unlink(name.c_str());
    }
};

extern "C" int rme2_abi_version() { return 1; }
extern "C" const char* rme2_error() { return last_error.c_str(); }
extern "C" int rme2_create(int rank, int device, int tokens, const char* name,
                           double timeout, int validate, int core, Rme2Proxy** out) {
    return api([&] {
        require(out != nullptr, "null output"); *out = nullptr;
        require(rank >= 0 && rank < 8 && device >= 0 && tokens > 0 && tokens <= 10000000 &&
                timeout > 0 && core >= -1 && core < CPU_SETSIZE, "invalid native configuration");
        require(name && name[0] == '/' && std::strlen(name) < 200 &&
                std::strchr(name + 1, '/') == nullptr, "invalid shared memory name");
        auto* p = new Rme2Proxy(rank, device, tokens, name, timeout, validate != 0, core);
        try { p->initialize(); } catch (...) { delete p; throw; }
        *out = p;
    });
}
extern "C" int rme2_export(Rme2Proxy* p, void* output) {
    return api([&] {
        require(p && output, "null export argument");
        check(cudaSetDevice(p->device), "select export GPU");
        cudaIpcMemHandle_t h{};
        static_assert(sizeof(h) == 64, "IPC ABI changed");
        check(cudaIpcGetMemHandle(&h, p->inbox), "export IPC inbox");
        std::memcpy(output, &h, sizeof(h));
    });
}
extern "C" int rme2_connect(Rme2Proxy* p, const void* handles) {
    return api([&] {
        require(p && handles && !p->connected, "invalid connect state");
        check(cudaSetDevice(p->device), "select connect GPU");
        for (int peer = 0; peer < 8; ++peer) {
            if (peer == p->rank) { p->peers.ptr[peer] = p->inbox; continue; }
            cudaIpcMemHandle_t h;
            std::memcpy(&h, static_cast<const char*>(handles) + peer * sizeof(h), sizeof(h));
            check(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&p->peers.ptr[peer]), h,
                                       cudaIpcMemLazyEnablePeerAccess), "open peer GPU inbox");
            p->opened[peer] = true;
        }
        p->connected = true;
        p->worker = std::thread(&Rme2Proxy::run, p);
    });
}
extern "C" int rme2_unlink(Rme2Proxy* p) {
    return api([&] {
        require(p && p->owner, "only creator can unlink");
        require(shm_unlink(p->name.c_str()) == 0, "unlink shared control failed");
        p->owner = false;
    });
}
extern "C" int rme2_disconnect(Rme2Proxy* p) {
    return api([&] {
        require(p && p->connected, "invalid disconnect state");
        {
            std::lock_guard<std::mutex> lock(p->mutex);
            require(!p->pending && !p->done, "disconnect requires completed wait");
        }
        p->stopping.store(true); p->changed.notify_all();
        if (p->worker.joinable()) p->worker.join();
        check(cudaSetDevice(p->device), "select disconnect GPU");
        for (int peer = 0; peer < 8; ++peer) {
            if (p->opened[peer]) {
                check(cudaIpcCloseMemHandle(p->peers.ptr[peer]), "close peer mapping");
                p->opened[peer] = false;
            }
        }
        p->connected = false;
    });
}
extern "C" int rme2_submit(Rme2Proxy* p, uint64_t iteration, const int32_t* offsets,
                           const int32_t* indices, void* stream) {
    return api([&] {
        require(p && offsets && indices && p->connected, "invalid submit arguments");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(p->failure.empty(), p->failure.c_str());
        require(!p->pending && !p->done && iteration == p->sequence + 1 && iteration != 0,
                "submit requires prior wait and consecutive iterations starting at 1");
        check(cudaSetDevice(p->device), "select submit GPU");
        p->submitted = Clock::now();
        auto s = static_cast<cudaStream_t>(stream);
        check(cudaEventRecord(p->begin, s), "record exchange start");
        launch_exchange(p->peers, p->rank, p->capacity, offsets, indices, s);
        check(cudaGetLastError(), "launch metadata exchange");
        check(cudaEventRecord(p->end, s), "record exchange end");
        p->local_values = indices; p->sequence = iteration; p->pending = true;
        p->changed.notify_all();
    });
}
extern "C" int rme2_wait(Rme2Proxy* p, Rme2Stats* stats) {
    return api([&] {
        require(p && stats, "null wait argument");
        std::unique_lock<std::mutex> lock(p->mutex);
        require(p->pending || p->done, "wait without submission");
        p->changed.wait(lock, [&] { return p->done; });
        require(p->failure.empty(), p->failure.c_str());
        *stats = p->stats; p->done = false;
    });
}
extern "C" int rme2_view(Rme2Proxy* p, int source, const int32_t** offsets,
                         const int32_t** indices, int* count) {
    return api([&] {
        require(p && source >= 0 && source < 8 && offsets && indices && count, "invalid view arguments");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(!p->pending && !p->done && p->failure.empty() && p->sequence > 0,
                "view requires successful wait");
        const auto& view = p->installed[source];
        *offsets = view.offsets; *indices = view.indices; *count = view.count;
    });
}
extern "C" void rme2_destroy(Rme2Proxy* p) { delete p; }
