#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

constexpr uint64_t kMagic = 0x49544552444f4e45ULL;  // "ITERDONE"
constexpr uint32_t kVersion = 1;
constexpr uint32_t kIterationDone = 1U << 1;

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

uint64_t load_acquire(const uint64_t* value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void store_release(uint64_t* value, uint64_t desired) {
    __atomic_store_n(value, desired, __ATOMIC_RELEASE);
}

uint64_t fetch_add_acq_rel(uint64_t* value, uint64_t increment) {
    return __atomic_fetch_add(value, increment, __ATOMIC_ACQ_REL);
}

uint64_t steady_nanoseconds() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct alignas(64) Notification {
    uint64_t sequence{0};
    uint64_t iteration{0};
    int32_t source_gpu{-1};
    int32_t destination_gpu{-1};
    uint32_t flags{0};
    uint32_t reserved{0};
    char padding[32]{};
};

struct alignas(64) SharedQueue {
    int32_t source_gpu{-1};
    uint32_t capacity{0};
    uint64_t head{0};
    uint64_t tail{0};
    uint64_t dropped{0};
    uint64_t sender_done{0};
    char padding[24]{};
};

struct alignas(64) SegmentHeader {
    uint64_t magic{0};
    uint32_t version{0};
    int32_t destination_gpu{-1};
    uint32_t proxy_count{0};
    uint32_t queue_count{0};
    uint32_t queue_capacity{0};
    uint32_t header_bytes{0};
    uint32_t queue_bytes{0};
    uint32_t notification_bytes{0};
    char padding[24]{};
};

struct alignas(64) SharedBarrier {
    uint64_t arrivals{0};
    uint64_t generation{0};
    char padding[48]{};
};

struct alignas(64) SharedControl {
    SharedBarrier iteration_barrier;
    SharedBarrier shutdown_barrier;
    uint64_t start{0};
    uint64_t abort{0};
    char padding[48]{};
};

struct Measurement {
    uint64_t total_ns{0};
    uint64_t enqueue_ns{0};
    uint64_t wait_ns{0};
};

static_assert(sizeof(Notification) == 64);
static_assert(sizeof(SharedQueue) == 64);
static_assert(sizeof(SegmentHeader) == 64);
static_assert(sizeof(SharedBarrier) == 64);
static_assert(sizeof(SharedControl) == 192);

struct Options {
    std::size_t proxies{8};
    std::size_t warmup{2000};
    std::size_t iterations{100000};
    std::size_t queue_depth{1024};
    uint64_t timeout_ms{10000};
    bool auxiliary_threads{true};
    std::string cpu_affinity;
};

std::size_t parse_size(const std::string& text, const std::string& option) {
    std::size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + option + ": " + text);
    }
    if (consumed != text.size() || value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("invalid value for " + option + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

bool parse_bool(std::string text, const std::string& option) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (text == "1" || text == "true" || text == "yes" || text == "on") return true;
    if (text == "0" || text == "false" || text == "no" || text == "off") return false;
    throw std::runtime_error("invalid Boolean value for " + option + ": " + text);
}

void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --proxies=N           proxy processes (default: 8)\n"
        << "  --warmup=N            untimed iterations (default: 2000)\n"
        << "  --iterations=N        measured iterations (default: 100000)\n"
        << "  --queue-depth=N       SPSC queue capacity (default: 1024)\n"
        << "  --timeout-ms=N        iteration timeout (default: 10000)\n"
        << "  --aux-threads=BOOL    keep forwarding/ready threads (default: true)\n"
        << "  --cpu-affinity=LIST   Linux process affinity inherited by threads\n"
        << "  --help                show this help\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            print_help(argv[0]);
            std::exit(0);
        }
        const auto equals = argument.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("expected --option=value, got: " + argument);
        }
        const auto key = argument.substr(0, equals);
        const auto value = argument.substr(equals + 1);
        if (key == "--proxies") options.proxies = parse_size(value, key);
        else if (key == "--warmup") options.warmup = parse_size(value, key);
        else if (key == "--iterations") options.iterations = parse_size(value, key);
        else if (key == "--queue-depth") options.queue_depth = parse_size(value, key);
        else if (key == "--timeout-ms") options.timeout_ms = parse_size(value, key);
        else if (key == "--aux-threads") options.auxiliary_threads = parse_bool(value, key);
        else if (key == "--cpu-affinity") options.cpu_affinity = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.proxies < 2 || options.proxies > 64) {
        throw std::runtime_error("--proxies must be in [2, 64]");
    }
    if (options.iterations == 0) throw std::runtime_error("--iterations must be > 0");
    if (options.queue_depth == 0 || options.queue_depth > UINT32_MAX) {
        throw std::runtime_error("--queue-depth must be in [1, UINT32_MAX]");
    }
    if (options.timeout_ms == 0) throw std::runtime_error("--timeout-ms must be > 0");
    return options;
}

std::vector<int> parse_cpu_list(const std::string& specification) {
    std::vector<int> cpus;
    std::stringstream stream(specification);
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (part.empty()) throw std::runtime_error("empty CPU-affinity element");
        const auto dash = part.find('-');
        if (dash == std::string::npos) {
            cpus.push_back(static_cast<int>(parse_size(part, "--cpu-affinity")));
            continue;
        }
        const int begin = static_cast<int>(parse_size(part.substr(0, dash), "--cpu-affinity"));
        const int end = static_cast<int>(parse_size(part.substr(dash + 1), "--cpu-affinity"));
        if (begin > end) throw std::runtime_error("descending CPU-affinity range: " + part);
        for (int cpu = begin; cpu <= end; ++cpu) cpus.push_back(cpu);
    }
    if (cpus.empty()) throw std::runtime_error("CPU-affinity list is empty");
    return cpus;
}

void apply_process_affinity(const std::string& specification) {
    if (specification.empty() || specification == "none") return;
#if defined(__linux__)
    const auto cpus = parse_cpu_list(specification);
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            throw std::runtime_error("CPU index outside CPU_SETSIZE");
        }
        CPU_SET(cpu, &set);
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        throw std::runtime_error("sched_setaffinity failed: " + std::string(std::strerror(errno)));
    }
#else
    (void)parse_cpu_list(specification);
    throw std::runtime_error("--cpu-affinity is supported only on Linux");
#endif
}

void barrier_wait(SharedBarrier* barrier, std::size_t participants, const uint64_t* abort) {
    const uint64_t generation = load_acquire(&barrier->generation);
    const uint64_t previous = fetch_add_acq_rel(&barrier->arrivals, 1);
    if (previous + 1 == participants) {
        store_release(&barrier->arrivals, 0);
        store_release(&barrier->generation, generation + 1);
        return;
    }
    while (load_acquire(&barrier->generation) == generation) {
        if (load_acquire(abort) != 0) throw std::runtime_error("benchmark aborted at process barrier");
        cpu_relax();
    }
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* description) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::runtime_error(std::string(description) + " size overflow");
    }
    return lhs * rhs;
}

std::size_t segment_bytes(std::size_t proxies, std::size_t capacity) {
    const std::size_t queues = proxies - 1;
    const std::size_t entries = checked_multiply(queues, capacity, "notification entry");
    return sizeof(SegmentHeader) + checked_multiply(queues, sizeof(SharedQueue), "queue") +
           checked_multiply(entries, sizeof(Notification), "notification segment");
}

SharedQueue* queue_at(SegmentHeader* header, std::size_t index) {
    auto* base = reinterpret_cast<char*>(header);
    return reinterpret_cast<SharedQueue*>(base + header->header_bytes + index * header->queue_bytes);
}

Notification* entries_at(SegmentHeader* header, std::size_t index) {
    auto* base = reinterpret_cast<char*>(header);
    const std::size_t queues_bytes = header->queue_count * header->queue_bytes;
    return reinterpret_cast<Notification*>(
        base + header->header_bytes + queues_bytes +
        index * header->queue_capacity * header->notification_bytes);
}

std::size_t source_queue_index(int source_gpu, int destination_gpu) {
    if (source_gpu == destination_gpu) throw std::runtime_error("self queue requested");
    return static_cast<std::size_t>(source_gpu < destination_gpu ? source_gpu : source_gpu - 1);
}

struct SegmentMapping {
    std::string name;
    SegmentHeader* header{nullptr};
    std::size_t bytes{0};
    int fd{-1};
};

struct DispatchState {
    std::mutex mutex;
    std::deque<Notification> pending;
};

class ProxySimulation {
public:
    ProxySimulation(
        int local_gpu,
        const Options& options,
        const std::vector<SegmentMapping>& segments,
        SharedControl* control,
        Measurement* measurements)
        : local_gpu_(local_gpu),
          options_(options),
          segments_(segments),
          control_(control),
          measurements_(measurements),
          done_generation_(new std::atomic<uint64_t>[options.proxies]) {
        for (std::size_t source = 0; source < options_.proxies; ++source) {
            done_generation_[source].store(0, std::memory_order_relaxed);
            if (static_cast<int>(source) != local_gpu_) destinations_.push_back(static_cast<int>(source));
        }
    }

    ~ProxySimulation() {
        if (!stopped_) {
            store_release(&control_->abort, 1);
            force_stop();
        }
    }

    void run() {
        apply_process_affinity(options_.cpu_affinity);
        dispatch_thread_ = std::thread(&ProxySimulation::dispatch_loop, this);
        receiver_thread_ = std::thread(&ProxySimulation::receiver_loop, this);
        if (options_.auxiliary_threads) {
            forwarding_thread_ = std::thread(&ProxySimulation::forwarding_loop, this);
            forwarding_ready_thread_ = std::thread(&ProxySimulation::forwarding_ready_loop, this);
        }

        const std::size_t total_iterations = options_.warmup + options_.iterations;
        for (std::size_t iteration = 0; iteration < total_iterations; ++iteration) {
            barrier_wait(&control_->iteration_barrier, options_.proxies, &control_->abort);
            const uint64_t begin = steady_nanoseconds();
            enqueue_iteration_done_notifications(iteration);
            const uint64_t after_enqueue = steady_nanoseconds();
            wait_for_iteration_done_notifications(iteration);
            const uint64_t end = steady_nanoseconds();
            if (iteration >= options_.warmup) {
                const std::size_t sample = iteration - options_.warmup;
                Measurement& result = measurements_[static_cast<std::size_t>(local_gpu_) *
                                                    options_.iterations + sample];
                result.total_ns = end - begin;
                result.enqueue_ns = after_enqueue - begin;
                result.wait_ns = end - after_enqueue;
            }
        }

        // No process starts teardown until every main thread has received the
        // final generation from every source.
        barrier_wait(&control_->shutdown_barrier, options_.proxies, &control_->abort);
        stop();
    }

private:
    void enqueue_iteration_done_notifications(uint64_t iteration) {
        std::vector<Notification> notifications;
        notifications.reserve(destinations_.size());
        for (const int destination : destinations_) {
            Notification notification;
            notification.iteration = iteration;
            notification.source_gpu = local_gpu_;
            notification.destination_gpu = destination;
            notification.flags = kIterationDone;
            notifications.push_back(notification);
        }
        std::lock_guard<std::mutex> lock(dispatch_.mutex);
        for (const auto& notification : notifications) dispatch_.pending.push_back(notification);
        notifications_enqueued_.fetch_add(notifications.size(), std::memory_order_relaxed);
    }

    void wait_for_iteration_done_notifications(uint64_t iteration) {
        const uint64_t generation = iteration + 1;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(options_.timeout_ms);
        while (true) {
            if (load_acquire(&control_->abort) != 0) {
                throw std::runtime_error("benchmark aborted while waiting for iteration-done notifications");
            }
            check_error();
            bool complete = true;
            for (std::size_t source = 0; source < options_.proxies; ++source) {
                if (static_cast<int>(source) == local_gpu_) continue;
                if (done_generation_[source].load(std::memory_order_acquire) < generation) {
                    complete = false;
                    break;
                }
            }
            if (complete) return;
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error("timed out waiting for iteration-done generation " +
                                         std::to_string(generation));
            }
            cpu_relax();
        }
    }

    void publish(Notification notification) {
        const auto destination = static_cast<std::size_t>(notification.destination_gpu);
        SegmentHeader* header = segments_.at(destination).header;
        const std::size_t queue_index = source_queue_index(local_gpu_, notification.destination_gpu);
        SharedQueue* queue = queue_at(header, queue_index);
        Notification* entries = entries_at(header, queue_index);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(options_.timeout_ms);
        while (true) {
            if (load_acquire(&control_->abort) != 0) {
                throw std::runtime_error("benchmark aborted while publishing notification");
            }
            const uint64_t head = load_acquire(&queue->head);
            const uint64_t tail = load_acquire(&queue->tail);
            if (head - tail < queue->capacity) {
                notification.sequence = head + 1;
                entries[head % queue->capacity] = notification;
                __atomic_thread_fence(__ATOMIC_RELEASE);
                store_release(&queue->head, head + 1);
                notifications_sent_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                fetch_add_acq_rel(&queue->dropped, 1);
                throw std::runtime_error("timed out waiting for shared notification queue space");
            }
            cpu_relax();
        }
    }

    void dispatch_loop() noexcept {
        try {
            while (true) {
                Notification notification;
                bool available = false;
                {
                    std::lock_guard<std::mutex> lock(dispatch_.mutex);
                    if (!dispatch_.pending.empty()) {
                        notification = dispatch_.pending.front();
                        dispatch_.pending.pop_front();
                        available = true;
                    }
                }
                if (available) {
                    publish(notification);
                    continue;
                }
                if (dispatch_stop_.load(std::memory_order_acquire)) {
                    mark_senders_done();
                    return;
                }
                cpu_relax();
            }
        } catch (const std::exception& error) {
            set_error(error.what());
        }
    }

    void drain_queue(std::size_t queue_index) {
        SegmentHeader* header = segments_.at(static_cast<std::size_t>(local_gpu_)).header;
        SharedQueue* queue = queue_at(header, queue_index);
        Notification* entries = entries_at(header, queue_index);
        while (true) {
            const uint64_t tail = load_acquire(&queue->tail);
            const uint64_t head = load_acquire(&queue->head);
            if (tail >= head) return;
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            const Notification notification = entries[tail % queue->capacity];
            if (notification.source_gpu != queue->source_gpu ||
                notification.destination_gpu != local_gpu_ ||
                (notification.flags & kIterationDone) == 0) {
                throw std::runtime_error("invalid iteration-done notification");
            }
            store_release(&queue->tail, tail + 1);
            const uint64_t generation = notification.iteration + 1;
            auto& done = done_generation_[static_cast<std::size_t>(notification.source_gpu)];
            const uint64_t previous = done.load(std::memory_order_acquire);
            if (previous >= generation) {
                throw std::runtime_error("duplicate or out-of-order iteration-done notification");
            }
            done.store(generation, std::memory_order_release);
            notifications_received_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    bool inbound_queues_complete() const {
        SegmentHeader* header = segments_.at(static_cast<std::size_t>(local_gpu_)).header;
        for (std::size_t index = 0; index < header->queue_count; ++index) {
            SharedQueue* queue = queue_at(header, index);
            if (load_acquire(&queue->sender_done) == 0 ||
                load_acquire(&queue->head) != load_acquire(&queue->tail)) {
                return false;
            }
        }
        return true;
    }

    void receiver_loop() noexcept {
        try {
            SegmentHeader* header = segments_.at(static_cast<std::size_t>(local_gpu_)).header;
            while (true) {
                bool progressed = false;
                for (std::size_t index = 0; index < header->queue_count; ++index) {
                    SharedQueue* queue = queue_at(header, index);
                    const uint64_t before = load_acquire(&queue->tail);
                    drain_queue(index);
                    progressed = progressed || load_acquire(&queue->tail) != before;
                }
                if (load_acquire(&control_->abort) != 0) return;
                if (receiver_stop_.load(std::memory_order_acquire) && inbound_queues_complete()) return;
                if (!progressed) cpu_relax();
            }
        } catch (const std::exception& error) {
            set_error(error.what());
        }
    }

    void forwarding_loop() {
        while (!forwarding_stop_.load(std::memory_order_acquire)) cpu_relax();
    }

    void forwarding_ready_loop() {
        while (!forwarding_stop_.load(std::memory_order_acquire)) cpu_relax();
    }

    void mark_senders_done() {
        for (const int destination : destinations_) {
            SegmentHeader* header = segments_.at(static_cast<std::size_t>(destination)).header;
            SharedQueue* queue = queue_at(header, source_queue_index(local_gpu_, destination));
            store_release(&queue->sender_done, 1);
        }
    }

    void set_error(const std::string& error) {
        std::lock_guard<std::mutex> lock(forwarding_mutex_);
        if (forwarding_error_.empty()) forwarding_error_ = error;
    }

    void check_error() {
        std::lock_guard<std::mutex> lock(forwarding_mutex_);
        if (!forwarding_error_.empty()) throw std::runtime_error(forwarding_error_);
    }

    void stop() {
        forwarding_stop_.store(true, std::memory_order_release);
        if (forwarding_ready_thread_.joinable()) forwarding_ready_thread_.join();
        if (forwarding_thread_.joinable()) forwarding_thread_.join();

        dispatch_stop_.store(true, std::memory_order_release);
        if (dispatch_thread_.joinable()) dispatch_thread_.join();

        receiver_stop_.store(true, std::memory_order_release);
        if (receiver_thread_.joinable()) receiver_thread_.join();
        stopped_ = true;
        check_error();

        const uint64_t expected = options_.proxies == 0 ? 0 :
            (options_.warmup + options_.iterations) * (options_.proxies - 1);
        if (notifications_enqueued_.load() != expected ||
            notifications_sent_.load() != expected ||
            notifications_received_.load() != expected) {
            throw std::runtime_error("notification accounting mismatch");
        }
    }

    void force_stop() noexcept {
        forwarding_stop_.store(true, std::memory_order_release);
        dispatch_stop_.store(true, std::memory_order_release);
        receiver_stop_.store(true, std::memory_order_release);
        if (forwarding_ready_thread_.joinable()) forwarding_ready_thread_.join();
        if (forwarding_thread_.joinable()) forwarding_thread_.join();
        if (dispatch_thread_.joinable()) dispatch_thread_.join();
        if (receiver_thread_.joinable()) receiver_thread_.join();
        stopped_ = true;
    }

    int local_gpu_;
    const Options& options_;
    const std::vector<SegmentMapping>& segments_;
    SharedControl* control_;
    Measurement* measurements_;
    std::vector<int> destinations_;
    std::unique_ptr<std::atomic<uint64_t>[]> done_generation_;
    DispatchState dispatch_;
    std::mutex forwarding_mutex_;
    std::string forwarding_error_;
    std::atomic<bool> forwarding_stop_{false};
    std::atomic<bool> dispatch_stop_{false};
    std::atomic<bool> receiver_stop_{false};
    std::atomic<uint64_t> notifications_enqueued_{0};
    std::atomic<uint64_t> notifications_sent_{0};
    std::atomic<uint64_t> notifications_received_{0};
    bool stopped_{false};
    std::thread forwarding_thread_;
    std::thread forwarding_ready_thread_;
    std::thread dispatch_thread_;
    std::thread receiver_thread_;
};

SegmentMapping create_segment(std::size_t destination, const Options& options) {
    SegmentMapping mapping;
    mapping.name = "/iteration_done_bench_" + std::to_string(getpid()) + "_" +
                   std::to_string(destination);
    mapping.bytes = segment_bytes(options.proxies, options.queue_depth);
    shm_unlink(mapping.name.c_str());
    mapping.fd = shm_open(mapping.name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (mapping.fd < 0) {
        throw std::runtime_error("shm_open failed for " + mapping.name + ": " + std::strerror(errno));
    }
    if (ftruncate(mapping.fd, static_cast<off_t>(mapping.bytes)) != 0) {
        throw std::runtime_error("ftruncate failed for " + mapping.name + ": " + std::strerror(errno));
    }
    void* memory = mmap(nullptr, mapping.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, mapping.fd, 0);
    if (memory == MAP_FAILED) {
        throw std::runtime_error("mmap failed for " + mapping.name + ": " + std::strerror(errno));
    }
    std::memset(memory, 0, mapping.bytes);
    mapping.header = static_cast<SegmentHeader*>(memory);
    mapping.header->magic = kMagic;
    mapping.header->version = kVersion;
    mapping.header->destination_gpu = static_cast<int32_t>(destination);
    mapping.header->proxy_count = static_cast<uint32_t>(options.proxies);
    mapping.header->queue_count = static_cast<uint32_t>(options.proxies - 1);
    mapping.header->queue_capacity = static_cast<uint32_t>(options.queue_depth);
    mapping.header->header_bytes = sizeof(SegmentHeader);
    mapping.header->queue_bytes = sizeof(SharedQueue);
    mapping.header->notification_bytes = sizeof(Notification);

    std::size_t queue_index = 0;
    for (std::size_t source = 0; source < options.proxies; ++source) {
        if (source == destination) continue;
        SharedQueue* queue = queue_at(mapping.header, queue_index++);
        queue->source_gpu = static_cast<int32_t>(source);
        queue->capacity = static_cast<uint32_t>(options.queue_depth);
    }
    return mapping;
}

void destroy_segments(std::vector<SegmentMapping>* segments) {
    for (auto& mapping : *segments) {
        if (mapping.header) munmap(mapping.header, mapping.bytes);
        if (mapping.fd >= 0) close(mapping.fd);
        if (!mapping.name.empty()) shm_unlink(mapping.name.c_str());
        mapping.header = nullptr;
        mapping.fd = -1;
    }
}

void* map_anonymous_shared(std::size_t bytes) {
    void* memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) throw std::runtime_error("anonymous shared mmap failed");
    std::memset(memory, 0, bytes);
    return memory;
}

double percentile(const std::vector<uint64_t>& sorted, double fraction) {
    if (sorted.empty()) return 0.0;
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return static_cast<double>(sorted[lower]) * (1.0 - weight) +
           static_cast<double>(sorted[upper]) * weight;
}

void print_distribution(const std::string& label, std::vector<uint64_t> samples) {
    std::sort(samples.begin(), samples.end());
    const long double sum = std::accumulate(samples.begin(), samples.end(), 0.0L);
    const double mean = samples.empty() ? 0.0 : static_cast<double>(sum / samples.size());
    auto us = [](double nanoseconds) { return nanoseconds / 1000.0; };
    std::cout << std::left << std::setw(18) << label
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(11) << us(mean)
              << std::setw(11) << us(percentile(samples, 0.50))
              << std::setw(11) << us(percentile(samples, 0.90))
              << std::setw(11) << us(percentile(samples, 0.95))
              << std::setw(11) << us(percentile(samples, 0.99))
              << std::setw(11) << us(percentile(samples, 0.999))
              << std::setw(11) << us(samples.empty() ? 0.0 : samples.back()) << '\n';
}

void report_results(const Options& options, const Measurement* measurements) {
    std::cout << "iteration-done CPU benchmark\n"
              << "proxies=" << options.proxies
              << " measured_iterations=" << options.iterations
              << " warmup_iterations=" << options.warmup
              << " queue_depth=" << options.queue_depth
              << " auxiliary_threads=" << (options.auxiliary_threads ? "true" : "false")
              << " cpu_affinity=" << (options.cpu_affinity.empty() ? "inherited" : options.cpu_affinity)
              << "\n\n"
              << std::left << std::setw(18) << "scope"
              << std::right << std::setw(11) << "mean_us"
              << std::setw(11) << "p50_us"
              << std::setw(11) << "p90_us"
              << std::setw(11) << "p95_us"
              << std::setw(11) << "p99_us"
              << std::setw(11) << "p99.9_us"
              << std::setw(11) << "max_us" << '\n';

    std::vector<uint64_t> aggregate_total;
    std::vector<uint64_t> aggregate_enqueue;
    std::vector<uint64_t> aggregate_wait;
    aggregate_total.reserve(options.proxies * options.iterations);
    aggregate_enqueue.reserve(options.proxies * options.iterations);
    aggregate_wait.reserve(options.proxies * options.iterations);
    for (std::size_t proxy = 0; proxy < options.proxies; ++proxy) {
        std::vector<uint64_t> proxy_total;
        proxy_total.reserve(options.iterations);
        for (std::size_t sample = 0; sample < options.iterations; ++sample) {
            const Measurement& result = measurements[proxy * options.iterations + sample];
            proxy_total.push_back(result.total_ns);
            aggregate_total.push_back(result.total_ns);
            aggregate_enqueue.push_back(result.enqueue_ns);
            aggregate_wait.push_back(result.wait_ns);
        }
        print_distribution("proxy" + std::to_string(proxy) + " total", std::move(proxy_total));
    }
    std::cout << '\n';
    print_distribution("aggregate total", std::move(aggregate_total));
    print_distribution("aggregate enqueue", std::move(aggregate_enqueue));
    print_distribution("aggregate wait", std::move(aggregate_wait));
}

int run_benchmark(const Options& options) {
    std::vector<SegmentMapping> segments;
    segments.reserve(options.proxies);
    try {
        for (std::size_t destination = 0; destination < options.proxies; ++destination) {
            segments.push_back(create_segment(destination, options));
        }
    } catch (...) {
        destroy_segments(&segments);
        throw;
    }

    auto* control = static_cast<SharedControl*>(map_anonymous_shared(sizeof(SharedControl)));
    const std::size_t result_count = checked_multiply(options.proxies, options.iterations, "result");
    const std::size_t result_bytes = checked_multiply(result_count, sizeof(Measurement), "result byte");
    auto* measurements = static_cast<Measurement*>(map_anonymous_shared(result_bytes));

    std::vector<pid_t> children;
    children.reserve(options.proxies);
    bool fork_failed = false;
    for (std::size_t proxy = 0; proxy < options.proxies; ++proxy) {
        const pid_t pid = fork();
        if (pid < 0) {
            fork_failed = true;
            break;
        }
        if (pid == 0) {
            while (load_acquire(&control->start) == 0 && load_acquire(&control->abort) == 0) cpu_relax();
            if (load_acquire(&control->abort) != 0) _exit(2);
            try {
                ProxySimulation simulation(
                    static_cast<int>(proxy), options, segments, control, measurements);
                simulation.run();
                _exit(0);
            } catch (const std::exception& error) {
                store_release(&control->abort, 1);
                std::cerr << "proxy " << proxy << " failed: " << error.what() << '\n';
                _exit(1);
            }
        }
        children.push_back(pid);
    }

    if (fork_failed) store_release(&control->abort, 1);
    else store_release(&control->start, 1);

    bool child_failed = fork_failed;
    for (const pid_t pid : children) {
        int status = 0;
        if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            child_failed = true;
        }
    }

    if (!child_failed) report_results(options, measurements);
    munmap(measurements, result_bytes);
    munmap(control, sizeof(SharedControl));
    destroy_segments(&segments);
    if (child_failed) throw std::runtime_error("one or more proxy processes failed");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run_benchmark(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "iteration_done_benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
