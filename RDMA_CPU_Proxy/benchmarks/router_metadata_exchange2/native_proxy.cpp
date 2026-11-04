#include "native_proxy.h"
#include "exchange.cuh"
#include "metadata_view.hpp"
#include "readback_progress.hpp"
#include "epoch_barrier.hpp"
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
struct Shared { Slot slots[rme2::gpus]; Slot metadata_arrivals[rme2::gpus]; Slot launch_arrivals[rme2::gpus]; Slot aborted; };
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr), "64-bit atomics required");
uint64_t acquire(const uint64_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
void release(uint64_t* p, uint64_t value) { __atomic_store_n(p, value, __ATOMIC_RELEASE); }
}

struct Rme2Proxy {
    int rank, device, tokens, capacity, core;
    double timeout;
    bool validate, per_source_readback, metadata_barrier, owner = false, connected = false, pending = false, done = false;
    std::string name, failure;
    Shared* shared = nullptr;
    int32_t* inbox = nullptr;
    int32_t* pinned = nullptr;
    PeerArenas peers{};
    std::array<bool, rme2::gpus> opened{};
    cudaStream_t read_stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr, read_done = nullptr, metadata_done = nullptr;
    std::array<cudaStream_t, rme2::gpus> source_streams{};
    std::array<cudaEvent_t, rme2::gpus> source_events{};
    cudaGraphExec_t metadata_graph = nullptr;
    cudaStream_t capture_stream = nullptr, work_stream = nullptr;
    bool capturing = false, launch_barrier = false;
    cudaEvent_t metadata_begin = nullptr, metadata_end = nullptr;
    const int32_t* local_offsets = nullptr;
    std::array<rme2::Installed, rme2::gpus> installed{};
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<bool> stopping{false};
    uint64_t sequence = 0;
    const int32_t* local_values = nullptr;
    Clock::time_point submitted;
    Rme2Stats stats{};

    Rme2Proxy(int r, int d, int t, const char* n, double seconds, bool v, int c, bool progressive, bool barrier)
        : rank(r), device(d), tokens(t), capacity(t * 8), core(c), timeout(seconds),
          validate(v), per_source_readback(progressive), metadata_barrier(barrier), name(n) {}

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
        if (metadata_barrier)
            check(cudaEventCreateWithFlags(&metadata_done, cudaEventDisableTiming), "create metadata completion event");
        if (per_source_readback) {
            for (int s = 0; s < rme2::gpus; ++s) {
                check(cudaStreamCreateWithFlags(&source_streams[s], cudaStreamNonBlocking),
                      "create per-source readback stream");
                check(cudaEventCreateWithFlags(&source_events[s], cudaEventDisableTiming),
                      "create per-source readback event");
            }
        }
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
    bool event_ready(cudaEvent_t event) {
        const auto status = cudaEventQuery(event);
        if (status == cudaErrorNotReady) return false;
        check(status, "query progressive readback event");
        return true;
    }
    void process_per_source() {
        std::array<rme2::ReadbackStage, rme2::gpus> stages{};
        std::array<Clock::time_point, rme2::gpus> header_start{}, payload_start{};
        const auto started = Clock::now();
        bool published = false;
        int observed = 0, installed_count = 0;
        while (installed_count != rme2::gpus) {
            check_deadline(started);
            // Never wait for our own GPU before progressing already-ready peers.
            // Still publish promptly so peers cannot deadlock behind our readback.
            if (!published && event_ready(end)) {
                stats.sender_completion_wait_ms = ms(started);
                release(&shared->slots[rank].complete, sequence);
                published = true;
            }
            bool changed = rme2::progress_sources(stages,
                [&](size_t s) { return acquire(&shared->slots[s].complete) >= sequence; },
                [&](size_t s) {
                    if (++observed == rme2::gpus)
                        stats.peer_completion_wait_ms = ms(started);
                    header_start[s] = Clock::now();
                    const size_t bytes = rme2::header_words * sizeof(int32_t);
                    check(cudaMemcpyAsync(pinned + s * rme2::header_words,
                                          inbox + s * rme2::header_words, bytes,
                                          cudaMemcpyDeviceToHost, source_streams[s]), "copy ready source header");
                    check(cudaEventRecord(source_events[s], source_streams[s]), "record source header completion");
                    stats.header_bytes += bytes;
                },
                [&](size_t s) { return event_ready(source_events[s]); },
                [&](size_t s) {
                    stats.headers_d2h_ms += ms(header_start[s]);
                    payload_start[s] = Clock::now();
                    const int32_t* header = pinned + s * rme2::header_words;
                    const int count = rme2::check_header(header, capacity);
                    const size_t bytes = size_t(count) * sizeof(int32_t);
                    const int32_t* src = int(s) == rank ? local_values + header[0] : payload(inbox, s);
                    if (bytes) check(cudaMemcpyAsync(payload(pinned, s), src, bytes,
                                                    cudaMemcpyDeviceToHost, source_streams[s]), "copy ready source payload");
                    // Even empty payloads use an event, keeping lifetime and
                    // completion semantics identical without a zero-byte copy.
                    check(cudaEventRecord(source_events[s], source_streams[s]), "record source payload completion");
                    stats.received_index_bytes += bytes;
                    if (int(s) != rank) stats.peer_received_index_bytes += bytes;
                },
                [&](size_t s) { return event_ready(source_events[s]); },
                [&](size_t s) {
                    stats.payload_d2h_ms += ms(payload_start[s]);
                    const auto install_start = Clock::now();
                    rme2::install(installed[s], pinned + s * rme2::header_words,
                                  payload(pinned, s), tokens, sequence, validate);
                    stats.install_ms += ms(install_start);
                    ++installed_count;
                });
            if (!changed) std::this_thread::yield();
        }
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, begin, end), "exchange event timing");
        stats.exchange_gpu_ms = elapsed;
        stats.native_ms = ms(submitted);
    }
    void process() {
        if (per_source_readback) { process_per_source(); return; }
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
    void submit_exchange(cudaStream_t s) {
        if (metadata_barrier) {
            try {
                // Metadata has already been queued on this stream. Only
                // this event's completion permits arrival at the CPU barrier.
                auto started = Clock::now();
                check(cudaEventRecord(metadata_done, s), "record metadata completion");
                wait_event(metadata_done);
                stats.metadata_completion_wait_ms = ms(started);
                started = Clock::now();
                release(&shared->metadata_arrivals[rank].complete, sequence);
                rme2::wait_epoch_all(rme2::gpus, sequence,
                    [&](int peer) { return acquire(&shared->metadata_arrivals[peer].complete); },
                    [&] {
                        check_deadline(started);
                        std::this_thread::yield();
                    });
                stats.metadata_barrier_wait_ms = ms(started);
            } catch (...) {
                release(&shared->aborted.complete, 1);
                throw;
            }
        }
        check(cudaEventRecord(begin, s), "record exchange start");
        launch_exchange(peers, rank, capacity, local_offsets, local_values, s);
        check(cudaGetLastError(), "launch metadata exchange");
        check(cudaEventRecord(end, s), "record exchange end");
    }
    void launch_metadata() {
        if (launch_barrier) {
            const auto start = Clock::now();
            release(&shared->launch_arrivals[rank].complete, sequence);
            rme2::wait_epoch_all(rme2::gpus, sequence,
                [&](int peer) { return acquire(&shared->launch_arrivals[peer].complete); },
                [&] { check_deadline(start); std::this_thread::yield(); });
            stats.metadata_launch_barrier_ms = ms(start);
        }
        check(cudaEventRecord(metadata_begin, work_stream), "record native metadata start");
        check(cudaGraphLaunch(metadata_graph, work_stream), "launch native metadata graph");
        check(cudaEventRecord(metadata_end, work_stream), "record native metadata end");
        submit_exchange(work_stream);
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
                if (metadata_graph) launch_metadata();
                process();
                if (metadata_graph) {
                    float elapsed = 0;
                    check(cudaEventElapsedTime(&elapsed, metadata_begin, metadata_end), "native metadata timing");
                    stats.metadata_gpu_ms = elapsed;
                    stats.native_ms = ms(submitted);
                }
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
        if (capturing) {
            cudaGraph_t discarded = nullptr;
            cudaStreamEndCapture(capture_stream, &discarded);
            if (discarded) cudaGraphDestroy(discarded);
        }
        if (metadata_graph && work_stream) cudaStreamSynchronize(work_stream);
        if (read_stream) cudaStreamSynchronize(read_stream);
        for (auto stream : source_streams) if (stream) cudaStreamSynchronize(stream);
        if (end) cudaEventSynchronize(end);
        for (int p = 0; p < rme2::gpus; ++p)
            if (opened[p]) cudaIpcCloseMemHandle(peers.ptr[p]);
        if (read_stream) cudaStreamDestroy(read_stream);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (read_done) cudaEventDestroy(read_done);
        if (metadata_done) cudaEventDestroy(metadata_done);
        for (auto event : source_events) if (event) cudaEventDestroy(event);
        for (auto stream : source_streams) if (stream) cudaStreamDestroy(stream);
        if (metadata_graph) cudaGraphExecDestroy(metadata_graph);
        if (metadata_begin) cudaEventDestroy(metadata_begin);
        if (metadata_end) cudaEventDestroy(metadata_end);
        if (inbox) cudaFree(inbox);
        if (pinned) cudaFreeHost(pinned);
        if (shared) munmap(shared, sizeof(Shared));
        if (owner) shm_unlink(name.c_str());
    }
};

extern "C" int rme2_abi_version() { return 4; }
extern "C" const char* rme2_error() { return last_error.c_str(); }
extern "C" int rme2_create(int rank, int device, int tokens, const char* name,
                           double timeout, int validate, int core, int per_source_readback, int metadata_barrier, Rme2Proxy** out) {
    return api([&] {
        require(out != nullptr, "null output"); *out = nullptr;
        require(rank >= 0 && rank < 8 && device >= 0 && tokens > 0 && tokens <= 10000000 &&
                timeout > 0 && core >= -1 && core < CPU_SETSIZE, "invalid native configuration");
        require(name && name[0] == '/' && std::strlen(name) < 200 &&
                std::strchr(name + 1, '/') == nullptr, "invalid shared memory name");
        require(per_source_readback == 0 || per_source_readback == 1, "invalid readback mode");
        require(metadata_barrier == 0 || metadata_barrier == 1, "invalid metadata barrier mode");
        auto* p = new Rme2Proxy(rank, device, tokens, name, timeout, validate != 0, core, per_source_readback != 0, metadata_barrier != 0);
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
extern "C" int rme2_capture_begin(Rme2Proxy* p, void* stream) {
    return api([&] {
        require(p && p->connected, "capture requires connected proxy");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(!p->pending && !p->done && !p->capturing && !p->metadata_graph && p->sequence == 0,
                "capture must precede all submissions and occur only once");
        require(p->failure.empty(), p->failure.c_str());
        check(cudaSetDevice(p->device), "select capture GPU");
        check(cudaEventCreate(&p->metadata_begin), "create native metadata start");
        check(cudaEventCreate(&p->metadata_end), "create native metadata end");
        p->capture_stream = static_cast<cudaStream_t>(stream);
        check(cudaStreamBeginCapture(p->capture_stream, cudaStreamCaptureModeThreadLocal), "begin metadata capture");
        p->capturing = true;
    });
}
extern "C" int rme2_capture_end(Rme2Proxy* p, int launch_barrier) {
    return api([&] {
        require(p && p->capturing, "no active metadata capture");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(launch_barrier == 0 || launch_barrier == 1, "invalid launch barrier mode");
        check(cudaSetDevice(p->device), "select capture end GPU");
        cudaGraph_t graph = nullptr;
        auto status = cudaStreamEndCapture(p->capture_stream, &graph);
        p->capturing = false;
        check(status, "end metadata capture");
        try {
            check(cudaGraphInstantiate(&p->metadata_graph, graph, nullptr, nullptr, 0), "instantiate native metadata graph");
            // Move upload/first launch costs outside measured iterations.
            check(cudaGraphUpload(p->metadata_graph, p->capture_stream), "upload native metadata graph");
            check(cudaStreamSynchronize(p->capture_stream), "complete native graph upload");
        } catch (...) { if (graph) cudaGraphDestroy(graph); throw; }
        check(cudaGraphDestroy(graph), "release captured graph definition");
        p->launch_barrier = launch_barrier != 0;
    });
}
extern "C" int rme2_submit(Rme2Proxy* p, uint64_t iteration, const int32_t* offsets,
                           const int32_t* indices, void* stream) {
    return api([&] {
        require(p && offsets && indices && p->connected, "invalid submit arguments");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(p->failure.empty(), p->failure.c_str());
        require(!p->pending && !p->done && !p->capturing && iteration == p->sequence + 1 && iteration != 0,
                "submit requires prior wait and consecutive iterations starting at 1");
        check(cudaSetDevice(p->device), "select submit GPU");
        p->submitted = Clock::now();
        p->stats = {};
        auto s = static_cast<cudaStream_t>(stream);
        p->local_values = indices; p->local_offsets = offsets;
        p->work_stream = s; p->sequence = iteration;
        if (!p->metadata_graph) p->submit_exchange(s);
        p->pending = true;
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
