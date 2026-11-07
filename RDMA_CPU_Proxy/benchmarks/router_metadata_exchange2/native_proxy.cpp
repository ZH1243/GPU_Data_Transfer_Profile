#include "native_proxy.h"
#include "exchange.cuh"
#include "metadata_view.hpp"
#include "readback_progress.hpp"
#include "epoch_barrier.hpp"
#include "payload_batch.hpp"
#include "rdma_transport.hpp"
#include <cuda.h>
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
    int nodes, node;
    std::unique_ptr<rme2::MetadataRdma> rdma;
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
    cudaGraphExec_t staging_graph = nullptr;
    cudaStream_t capture_stream = nullptr, work_stream = nullptr;
    bool capturing = false, capturing_staging = false, launch_barrier = false;
    cudaEvent_t metadata_begin = nullptr, metadata_end = nullptr;
    cudaEvent_t staging_begin = nullptr, staging_end = nullptr;
    const int32_t* local_offsets = nullptr;
    const int32_t* local_node_offsets = nullptr;
    const int32_t* local_x3 = nullptr;
    const uint8_t* local_x4 = nullptr;
    std::array<rme2::Installed, 16> installed{};
    rme2::two::InstalledX3X4 installed_x3_x4{};
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<bool> stopping{false};
    uint64_t sequence = 0;
    const int32_t* local_values = nullptr;
    Clock::time_point submitted;
    Rme2Stats stats{};
    bool batch_payload_copy = false;
    bool d2h_after_local_staging = false;
    bool local_staging_after_d2h = false;
    bool staging_readback_started = false, staging_readback_ready = false;
    Clock::time_point staging_readback_start;
    rme2::PayloadBatch payload_batch;

    Rme2Proxy(int r, int d, int t, const char* n, double seconds, bool v, int c, bool progressive, bool barrier,
              int ns, int nd)
        : rank(r), device(d), tokens(t), capacity(t * 8), core(c), nodes(ns), node(nd), timeout(seconds),
          validate(v), per_source_readback(progressive), metadata_barrier(barrier), name(n) {}

    size_t words() const {
        return nodes == 2 ? rme2::two::words(capacity) :
            size_t(rme2::gpus) * (rme2::header_words + size_t(capacity));
    }
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
    bool readback_ready() {
        // Poll on the CPU before starting D2H timers. A stream wait inside the
        // copy interval would incorrectly charge gather time to D2H latency.
        if (!d2h_after_local_staging || staging_readback_ready) return true;
        if (!staging_readback_started) {
            staging_readback_start = Clock::now();
            staging_readback_started = true;
        }
        if (!event_ready(staging_end)) return false;
        stats.staging_before_d2h_wait_ms = ms(staging_readback_start);
        staging_readback_ready = true;
        return true;
    }
    void wait_for_readback() {
        while (!readback_ready()) {
            check_deadline(submitted);
            std::this_thread::yield();
        }
    }
    void process_per_source() {
        std::array<rme2::ReadbackStage, rme2::gpus> stages{};
        std::array<Clock::time_point, rme2::gpus> header_start{}, payload_start{}, payload_wait_start{};
        const auto started = Clock::now();
        bool published = false;
        int observed = 0, installed_count = 0;
        while (installed_count != rme2::gpus) {
            check_deadline(started);
            // Publish promptly even when the optional staging gate delays
            // readback, so peers cannot deadlock behind our gather/readback.
            if (!published && event_ready(end)) {
                stats.sender_completion_wait_ms = ms(started);
                release(&shared->slots[rank].complete, sequence);
                published = true;
            }
            bool changed = rme2::progress_sources(stages,
                [&](size_t s) {
                    return acquire(&shared->slots[s].complete) >= sequence && readback_ready();
                },
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
                    const auto checks_start = Clock::now();
                    const int32_t* header = pinned + s * rme2::header_words;
                    const int count = rme2::check_header(header, capacity);
                    const size_t bytes = size_t(count) * sizeof(int32_t);
                    const int32_t* src = int(s) == rank ? local_values + header[0] : payload(inbox, s);
                    auto* dst = payload(pinned, s);
                    stats.payload_checks_ms += ms(checks_start);
                    if (bytes) {
                        const auto submit_start = Clock::now();
                        check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                              source_streams[s]), "copy ready source payload");
                        stats.payload_submit_ms += ms(submit_start);
                    }
                    // Even empty payloads use an event, keeping lifetime and
                    // completion semantics identical without a zero-byte copy.
                    const auto event_start = Clock::now();
                    check(cudaEventRecord(source_events[s], source_streams[s]), "record source payload completion");
                    stats.payload_event_record_ms += ms(event_start);
                    payload_wait_start[s] = Clock::now();
                    stats.received_index_bytes += bytes;
                    if (int(s) != rank) stats.peer_received_index_bytes += bytes;
                },
                [&](size_t s) {
                    if (!event_ready(source_events[s])) return false;
                    stats.payload_completion_ms += ms(payload_wait_start[s]);
                    return true;
                },
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
        // Every source's payload completion was observed; all D2H is drained.
        if (local_staging_after_d2h) launch_staging(work_stream);
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, begin, end), "exchange event timing");
        stats.exchange_gpu_ms = elapsed;
        finish_payload_breakdown();
        stats.native_ms = ms(submitted);
    }
    void finish_payload_breakdown() {
        // Uninstrumented loop/accounting work and timer boundary overhead. In
        // progressive mode the parent and completion fields sum source spans.
        stats.payload_other_ms = stats.payload_d2h_ms - stats.payload_checks_ms
            - stats.payload_submit_ms - stats.payload_event_record_ms - stats.payload_completion_ms;
    }
    void process_two() {
        namespace two = rme2::two;
        auto start = Clock::now();
        wait_event(end);
        stats.sender_completion_wait_ms = ms(start);
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, begin, end), "two-node exchange timing");
        stats.exchange_gpu_ms = elapsed;
        // Each producer fences ALL peer writes before its event. An idle event
        // on the relay alone would not prove the other GPUs finished writing.
        release(&shared->slots[rank].complete, sequence);
        start = Clock::now();
        rme2::wait_epoch_all(8, sequence,
            [&](int s) { return acquire(&shared->slots[s].complete); },
            [&] { check_deadline(start); std::this_thread::yield(); });
        stats.peer_completion_wait_ms = ms(start);
        wait_for_readback();
        start = Clock::now();
        stats.header_bytes = two::directory_words * sizeof(int32_t);
        check(cudaMemcpyAsync(pinned, inbox, stats.header_bytes, cudaMemcpyDeviceToHost, read_stream),
              "copy two-node directory");
        check(cudaEventRecord(read_done, read_stream), "record directory completion");
        wait_event(read_done);
        stats.headers_d2h_ms = ms(start);

        const auto payload_start = Clock::now();
        for (int s = 0; s < 8; ++s) {
            const auto before = Clock::now();
            const int slot = node * 8 + s;
            const int32_t* h = pinned + slot * two::header_words;
            const int count = two::check_record(h, capacity, sequence, slot, node * 8 + rank);
            auto* dst = pinned + two::record_offset(slot, capacity);
            auto* src = inbox + two::record_offset(slot, capacity);
            std::memcpy(dst, h, two::record_bytes(0));
            const size_t bytes = size_t(count) * 4;
            stats.payload_checks_ms += ms(before);
            if (bytes) {
                const auto submit_start = Clock::now();
                check(cudaMemcpyAsync(dst + two::header_words, src + two::header_words,
                                      bytes, cudaMemcpyDeviceToHost, read_stream), "copy local-node indices");
                stats.payload_submit_ms += ms(submit_start);
            }
            stats.received_index_bytes += bytes;
            if (s != rank) stats.peer_received_index_bytes += bytes;
        }
        start = Clock::now();
        check(cudaEventRecord(read_done, read_stream), "record local payload completion");
        stats.payload_event_record_ms = ms(start);
        const auto payload_wait_start = Clock::now();
        const auto rdma_start = Clock::now();
        for (int s = 0; s < 8; ++s) {
            const auto before = Clock::now();
            const int32_t* h = pinned + ((1 - node) * 8 + s) * two::header_words;
            const int count = two::check_record(h, capacity, sequence, node * 8 + s, (1 - node) * 8 + rank);
            rdma->post(sequence, s, count);
            stats.rdma_sent_bytes += two::record_bytes(count);
            stats.rdma_post_ms += ms(before);
        }
        {
            const auto before = Clock::now();
            const int32_t* h = pinned + two::sources * two::header_words;
            const int count = two::check_x3_x4_header(h, tokens, sequence,
                                                      node * 8 + rank, (1 - node) * 8 + rank);
            rdma->post_x3_x4(sequence, count);
            stats.x3_x4_sent_bytes = size_t(count) * 5;
            stats.rdma_sent_bytes += two::x3_x4_bytes(count);
            stats.rdma_post_ms += ms(before);
        }
        two::CompletionProgress progress(sequence);
        bool local_installed = false;
        while (!local_installed || !progress.complete()) {
            check_deadline(submitted);
            if (!local_installed && event_ready(read_done)) {
                stats.payload_completion_ms = ms(payload_wait_start);
                stats.payload_d2h_ms = ms(payload_start);
                finish_payload_breakdown();
                // Start gathering immediately after local D2H, without waiting
                // for remote metadata or outgoing RDMA completion.
                if (local_staging_after_d2h) launch_staging(work_stream);
                const auto before = Clock::now();
                for (int s = 0; s < 8; ++s) {
                    const int slot = node * 8 + s;
                    const int32_t* h = pinned + two::record_offset(slot, capacity);
                    rme2::install(installed[slot], h + 6, h + two::header_words,
                                  tokens, sequence, validate, two::epg);
                }
                stats.install_ms += ms(before);
                local_installed = true;
            }
            rdma->completions.clear();
            const auto before_poll = Clock::now();
            rdma->qp->poll(rdma->completions);
            stats.rdma_poll_ms += ms(before_poll);
            for (const auto& completion : rdma->completions) {
                if (completion.kind == rdma_proxy::CompletionKind::kSend) {
                    progress.complete_send(completion.wr_id);
                } else {
                    const int s = progress.receive_source(completion.wr_id, completion.imm_data);
                    const auto before = Clock::now();
                    if (s == two::x3_x4_tag) {
                        const int32_t* h = pinned + two::x3_x4_offset(capacity);
                        two::install_x3_x4(installed_x3_x4, h, tokens, sequence,
                                          (1 - node) * 8 + rank, node * 8 + rank,
                                          completion.byte_len, validate);
                        stats.x3_x4_received_bytes = size_t(installed_x3_x4.count) * 5;
                    } else {
                        const int slot = (1 - node) * 8 + s;
                        const int32_t* h = pinned + two::record_offset(slot, capacity);
                        two::check_received(h, capacity, sequence, slot, node * 8 + rank, completion.byte_len);
                        rme2::install(installed[slot], h + 6, h + two::header_words,
                                      tokens, sequence, validate, two::epg);
                        stats.received_index_bytes += size_t(installed[slot].count) * 4;
                        stats.peer_received_index_bytes += size_t(installed[slot].count) * 4;
                    }
                    stats.install_ms += ms(before);
                    stats.rdma_received_bytes += completion.byte_len;
                    progress.installed(s);
                    const auto repost_start = Clock::now();
                    rdma->qp->post_receive(sequence + 1);
                    stats.rdma_post_recv_ms += ms(repost_start);
                }
            }
            if (progress.complete() && stats.rdma_work_complete_ms == 0)
                stats.rdma_work_complete_ms = ms(rdma_start);
            if (rdma->completions.empty()) std::this_thread::yield();
        }
        stats.native_ms = ms(submitted);
    }
    void process() {
        if (nodes == 2) { process_two(); return; }
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
        wait_for_readback();
        start = Clock::now();
        stats.header_bytes = rme2::gpus * rme2::header_words * sizeof(int32_t);
        check(cudaMemcpyAsync(pinned, inbox, stats.header_bytes, cudaMemcpyDeviceToHost,
                              read_stream), "copy headers to pinned records");
        check(cudaEventRecord(read_done, read_stream), "record header readback");
        wait_event(read_done);
        stats.headers_d2h_ms = ms(start);
        start = Clock::now();
        if (batch_payload_copy) payload_batch.reset();
        for (int source = 0; source < rme2::gpus; ++source) {
            const auto checks_start = Clock::now();
            const int32_t* header = pinned + source * rme2::header_words;
            int count = rme2::check_header(header, capacity);
            auto bytes = size_t(count) * sizeof(int32_t);
            const int32_t* src = source == rank ? local_values + header[0] : payload(inbox, source);
            auto* dst = payload(pinned, source);
            if (batch_payload_copy) payload_batch.append(dst, src, bytes, source);
            stats.payload_checks_ms += ms(checks_start);
            if (bytes && !batch_payload_copy) {
                const auto submit_start = Clock::now();
                check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                      read_stream), "copy exact metadata payload");
                stats.payload_submit_ms += ms(submit_start);
            }
            stats.received_index_bytes += bytes;
            if (source != rank) stats.peer_received_index_bytes += bytes;
        }
        if (batch_payload_copy && payload_batch.count) {
#if CUDART_VERSION >= 12080
            cudaMemcpyAttributes attrs{};
            attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
            size_t attribute_index = 0;
            size_t failed_index = SIZE_MAX;
            const auto submit_start = Clock::now();
            const auto status = cudaMemcpyBatchAsync(
                payload_batch.destinations.data(), payload_batch.sources.data(),
                payload_batch.sizes.data(), payload_batch.count, &attrs,
                &attribute_index, 1, &failed_index, read_stream);
            stats.payload_submit_ms += ms(submit_start);
            if (status != cudaSuccess) {
                const std::string detail = failed_index < payload_batch.count
                    ? " at source " + std::to_string(payload_batch.source_ranks[failed_index])
                    : " (no specific batch entry)";
                throw std::runtime_error(std::string("cudaMemcpyBatchAsync payload") + detail + ": " + cudaGetErrorString(status));
            }
#else
            throw std::runtime_error("batch payload copy requires a CUDA 12.8+ build");
#endif
        }
        auto part_start = Clock::now();
        check(cudaEventRecord(read_done, read_stream), "record payload readback");
        stats.payload_event_record_ms = ms(part_start);
        part_start = Clock::now();
        wait_event(read_done);
        stats.payload_completion_ms = ms(part_start);
        stats.payload_d2h_ms = ms(start);
        finish_payload_breakdown();
        if (local_staging_after_d2h) launch_staging(work_stream);
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
        if (nodes == 2)
            launch_exchange_two(peers, rank, node, capacity, sequence, local_offsets, local_values,
                                local_node_offsets, local_x3, local_x4, s);
        else
            launch_exchange(peers, rank, capacity, local_offsets, local_values, s);
        check(cudaGetLastError(), "launch metadata exchange");
        // Publish readiness at the exchange boundary, BEFORE the gather. The
        // receiver can read metadata while this stream gathers activation rows.
        check(cudaEventRecord(end, s), "record exchange end");
        if (!local_staging_after_d2h) launch_staging(s);
    }
    void launch_staging(cudaStream_t s) {
        if (staging_graph) {
            check(cudaEventRecord(staging_begin, s), "record local staging start");
            check(cudaGraphLaunch(staging_graph, s), "launch local staging graph");
            check(cudaEventRecord(staging_end, s), "record local staging end");
        }
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
                // Drain any remaining gather tail. In serialized mode readback
                // already observed this event; publication never waits on it.
                if (staging_graph) {
                    const auto started = Clock::now();
                    wait_event(staging_end);
                    stats.local_staging_wait_ms = ms(started);
                    float elapsed = 0;
                    check(cudaEventElapsedTime(&elapsed, staging_begin, staging_end), "local staging timing");
                    stats.local_staging_gpu_ms = elapsed;
                }
                if (metadata_graph) {
                    float elapsed = 0;
                    check(cudaEventElapsedTime(&elapsed, metadata_begin, metadata_end), "native metadata timing");
                    stats.metadata_gpu_ms = elapsed;
                }
                stats.native_ms = ms(submitted);
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
        if ((metadata_graph || staging_graph) && work_stream) cudaStreamSynchronize(work_stream);
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
        if (staging_graph) cudaGraphExecDestroy(staging_graph);
        if (staging_begin) cudaEventDestroy(staging_begin);
        if (staging_end) cudaEventDestroy(staging_end);
        rdma.reset();
        if (inbox) cudaFree(inbox);
        if (pinned) cudaFreeHost(pinned);
        if (shared) munmap(shared, sizeof(Shared));
        if (owner) shm_unlink(name.c_str());
    }
};

extern "C" int rme2_abi_version() { return 11; }
extern "C" int rme2_set_local_staging_after_d2h(Rme2Proxy* p, int enabled) {
    return api([&] {
        require(p && (enabled == 0 || enabled == 1), "invalid deferred staging option");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(p->sequence == 0 && !p->pending && !p->done, "configure staging order before iterations");
        require(!enabled || !p->d2h_after_local_staging, "staging ordering options are mutually exclusive");
        p->local_staging_after_d2h = enabled != 0;
    });
}
extern "C" int rme2_set_d2h_after_local_staging(Rme2Proxy* p, int enabled) {
    return api([&] {
        require(p && (enabled == 0 || enabled == 1), "invalid staging readback option");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(p->sequence == 0 && !p->pending && !p->done, "configure staging readback before iterations");
        require(!enabled || !p->local_staging_after_d2h, "staging ordering options are mutually exclusive");
        p->d2h_after_local_staging = enabled != 0;
    });
}
extern "C" int rme2_set_batch_payload_copy(Rme2Proxy* p, int enabled) {
    return api([&] {
        require(p && (enabled == 0 || enabled == 1), "invalid batch payload option");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(p->sequence == 0 && !p->pending && !p->done, "configure payload copy before iterations");
        require(!enabled || !p->per_source_readback, "batch payload copy cannot combine with per-source readback");
        require(!enabled || p->nodes == 1, "two-node mode currently uses ordinary batched readback");
#if CUDART_VERSION < 12080
        require(!enabled, "batch payload copy requires a CUDA 12.8+ build (use CUDA 12.9 on Hopper)");
#endif
        p->batch_payload_copy = enabled != 0;
    });
}
extern "C" const char* rme2_error() { return last_error.c_str(); }
extern "C" int rme2_create(int rank, int device, int tokens, const char* name,
                           double timeout, int validate, int core, int per_source_readback, int metadata_barrier,
                           int nodes, int node, Rme2Proxy** out) {
    return api([&] {
        require(out != nullptr, "null output"); *out = nullptr;
        require(rank >= 0 && rank < 8 && device >= 0 && tokens > 0 && tokens <= 10000000 &&
                timeout > 0 && core >= -1 && core < CPU_SETSIZE, "invalid native configuration");
        require(name && name[0] == '/' && std::strlen(name) < 200 &&
                std::strchr(name + 1, '/') == nullptr, "invalid shared memory name");
        require(per_source_readback == 0 || per_source_readback == 1, "invalid readback mode");
        require(metadata_barrier == 0 || metadata_barrier == 1, "invalid metadata barrier mode");
        require((nodes == 1 || nodes == 2) && node >= 0 && node < nodes, "invalid node topology");
        require(nodes == 1 || !per_source_readback, "two-node mode requires all-senders readback");
        auto* p = new Rme2Proxy(rank, device, tokens, name, timeout, validate != 0, core,
                              per_source_readback != 0, metadata_barrier != 0, nodes, node);
        try { p->initialize(); } catch (...) { delete p; throw; }
        *out = p;
    });
}
extern "C" int rme2_rdma_init(Rme2Proxy* p, const char* device, int port, int gid,
                              char* descriptor, int descriptor_capacity) {
    return api([&] {
        require(p && p->nodes == 2 && !p->connected && !p->rdma && device && *device &&
                port >= 1 && port <= 255 && gid >= -1 && gid <= 255 && descriptor &&
                descriptor_capacity >= 4096, "invalid RDMA initialization");
#if !RDMA_PROXY_HAVE_VERBS
        throw std::runtime_error("two-node mode requires libibverbs; rebuild with -DRME2_REQUIRE_RDMA=ON");
#endif
        check(cudaSetDevice(p->device), "select RDMA owner GPU");
        unsigned int sync = 1;
        require(cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                      reinterpret_cast<CUdeviceptr>(p->inbox)) == CUDA_SUCCESS,
                "enable GPUDirect SYNC_MEMOPS on relay arena");
        p->rdma = std::make_unique<rme2::MetadataRdma>(p->node, p->rank, p->capacity,
                                                     device, port, gid, p->inbox, p->pinned);
        const auto data = p->rdma->descriptor();
        require(data.size() < size_t(descriptor_capacity), "RDMA descriptor exceeds output capacity");
        std::memcpy(descriptor, data.c_str(), data.size() + 1);
    });
}
extern "C" int rme2_rdma_connect(Rme2Proxy* p, const char* descriptor) {
    return api([&] {
        require(p && p->rdma && !p->connected && descriptor, "invalid RDMA connect state");
        p->rdma->connect(descriptor);
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
namespace {
int capture_begin(Rme2Proxy* p, void* stream, bool staging) {
    return api([&] {
        require(p && p->connected, "capture requires connected proxy");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(!p->pending && !p->done && !p->capturing &&
                !(staging ? p->staging_graph : p->metadata_graph) && p->sequence == 0,
                "each capture must precede all submissions and occur only once");
        require(p->failure.empty(), p->failure.c_str());
        check(cudaSetDevice(p->device), "select capture GPU");
        check(cudaEventCreate(staging ? &p->staging_begin : &p->metadata_begin), "create graph start event");
        check(cudaEventCreate(staging ? &p->staging_end : &p->metadata_end), "create graph end event");
        p->capture_stream = static_cast<cudaStream_t>(stream);
        check(cudaStreamBeginCapture(p->capture_stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
        p->capturing = true;
        p->capturing_staging = staging;
    });
}
int capture_end(Rme2Proxy* p, int launch_barrier, bool staging) {
    return api([&] {
        require(p && p->capturing && p->capturing_staging == staging, "no matching active graph capture");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(launch_barrier == 0 || launch_barrier == 1, "invalid launch barrier mode");
        check(cudaSetDevice(p->device), "select capture end GPU");
        auto* executable = staging ? &p->staging_graph : &p->metadata_graph;
        cudaGraph_t graph = nullptr;
        auto status = cudaStreamEndCapture(p->capture_stream, &graph);
        p->capturing = false;
        check(status, "end graph capture");
        try {
            check(cudaGraphInstantiate(executable, graph, nullptr, nullptr, 0), "instantiate native graph");
            // Move upload/first launch costs outside measured iterations.
            check(cudaGraphUpload(*executable, p->capture_stream), "upload native graph");
            check(cudaStreamSynchronize(p->capture_stream), "complete native graph upload");
        } catch (...) { if (graph) cudaGraphDestroy(graph); throw; }
        check(cudaGraphDestroy(graph), "release captured graph definition");
        if (!staging) p->launch_barrier = launch_barrier != 0;
    });
}
} // namespace
extern "C" int rme2_capture_begin(Rme2Proxy* p, void* stream) { return capture_begin(p, stream, false); }
extern "C" int rme2_capture_end(Rme2Proxy* p, int launch_barrier) { return capture_end(p, launch_barrier, false); }
extern "C" int rme2_staging_capture_begin(Rme2Proxy* p, void* stream) { return capture_begin(p, stream, true); }
extern "C" int rme2_staging_capture_end(Rme2Proxy* p) { return capture_end(p, 0, true); }
extern "C" int rme2_submit(Rme2Proxy* p, uint64_t iteration, const int32_t* offsets,
                           const int32_t* indices, const int32_t* node_offsets,
                           const int32_t* x3, const uint8_t* x4, void* stream) {
    return api([&] {
        require(p && offsets && indices && p->connected, "invalid submit arguments");
        require(p->nodes == 1 || (node_offsets && x3 && x4), "two-node submit requires x3/x4 metadata");
        require(p->nodes == 1 || (p->rdma && p->rdma->connected), "RDMA setup is incomplete");
        if (p->nodes == 2) rme2::two::send_id(iteration, 0);
        std::lock_guard<std::mutex> lock(p->mutex);
        require(p->failure.empty(), p->failure.c_str());
        require(!p->pending && !p->done && !p->capturing && iteration == p->sequence + 1 && iteration != 0,
                "submit requires prior wait and consecutive iterations starting at 1");
        require(!(p->d2h_after_local_staging || p->local_staging_after_d2h) || p->staging_graph,
                "serialized staging/D2H requires a captured staging graph");
        check(cudaSetDevice(p->device), "select submit GPU");
        p->submitted = Clock::now();
        p->stats = {};
        p->staging_readback_started = p->staging_readback_ready = false;
        auto s = static_cast<cudaStream_t>(stream);
        p->local_values = indices; p->local_offsets = offsets;
        p->local_node_offsets = node_offsets; p->local_x3 = x3; p->local_x4 = x4;
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
        require(p && source >= 0 && source < p->nodes * 8 && offsets && indices && count, "invalid view arguments");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(!p->pending && !p->done && p->failure.empty() && p->sequence > 0,
                "view requires successful wait");
        const auto& view = p->installed[source];
        *offsets = view.offsets; *indices = view.indices; *count = view.count;
    });
}
extern "C" int rme2_x3_x4_view(Rme2Proxy* p, const int32_t** x3, const uint8_t** x4, int* count) {
    return api([&] {
        require(p && p->nodes == 2 && x3 && x4 && count, "invalid x3/x4 view arguments");
        std::lock_guard<std::mutex> lock(p->mutex);
        require(!p->pending && !p->done && p->failure.empty() && p->sequence > 0 &&
                p->installed_x3_x4.iteration == p->sequence, "x3/x4 view requires successful wait");
        *x3 = p->installed_x3_x4.x3; *x4 = p->installed_x3_x4.x4; *count = p->installed_x3_x4.count;
    });
}
extern "C" void rme2_destroy(Rme2Proxy* p) { delete p; }
