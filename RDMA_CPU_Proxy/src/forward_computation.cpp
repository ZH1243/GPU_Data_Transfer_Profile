#include "forward_computation.hpp"

#include "cuda_buffers.hpp"
#include "logging.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace rdma_proxy {

#if RDMA_PROXY_HAVE_CUDA
void launch_persistent_forward_computation_kernel(
    const ForwardDeviceQueueView* queues,
    uint32_t num_queues,
    uint32_t num_ctas,
    DataType dtype,
    bool load_only,
    ForwardQueueSignal* abort_signal,
    uint32_t* physical_sm_ids,
    std::size_t dynamic_shared_bytes,
    void* stream);
#endif

namespace {

uint64_t steady_nanoseconds_now() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint64_t load_acquire(const uint64_t* value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void store_release(uint64_t* value, uint64_t desired) {
    __atomic_store_n(value, desired, __ATOMIC_RELEASE);
}

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

float bf16_bits_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t float_to_bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

float fp16_bits_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exponent = (value >> 10) & 0x1fU;
    uint32_t mantissa = value & 0x3ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3ffU;
            const uint32_t fp32_exponent = static_cast<uint32_t>(127 - 15 - shift);
            bits = sign | (fp32_exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        exponent = exponent + (127U - 15U);
        bits = sign | (exponent << 23) | (mantissa << 13);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t float_to_fp16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U))) {
        ++rounded;
        if (rounded == 0x400U) {
            rounded = 0;
            ++exponent;
            if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | rounded);
}

float decode_value(uint16_t value, DataType dtype) {
    return dtype == DataType::kBF16 ? bf16_bits_to_float(value) : fp16_bits_to_float(value);
}

uint16_t encode_value(float value, DataType dtype) {
    return dtype == DataType::kBF16 ? float_to_bf16_bits(value) : float_to_fp16_bits(value);
}

#if RDMA_PROXY_HAVE_CUDA
void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

#endif

class CpuToGpuQueue {
public:
    CpuToGpuQueue(uint32_t id, uint32_t capacity, bool mock_mode)
        : id_(id), capacity_(capacity), mock_mode_(mock_mode) {}

    ~CpuToGpuQueue() { release(); }

    void initialize() {
        if (initialized_) return;
        if (mock_mode_) {
            allocate_mock(consumed_host_, sizeof(ForwardQueueSignal) * capacity_);
            consumed_device_ = consumed_host_;
            allocate_mock(tasks_host_, sizeof(ForwardComputeTask) * capacity_);
            tasks_device_ = tasks_host_;
            initialized_ = true;
            return;
        }
#if RDMA_PROXY_HAVE_CUDA
        allocate_host_staging(publication_staging_, sizeof(uint64_t) * capacity_);
        allocate_host_visible(
            consumed_host_, consumed_device_, sizeof(ForwardQueueSignal) * capacity_);
        allocate_host_staging(tasks_host_, sizeof(ForwardComputeTask) * capacity_);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&published_head_device_),
                              sizeof(uint64_t)),
                   "cudaMalloc CPU-to-GPU published head");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&tasks_device_),
                              sizeof(ForwardComputeTask) * capacity_),
                   "cudaMalloc CPU-to-GPU task ring");
        check_cuda(cudaMalloc(&dequeue_device_, sizeof(uint64_t)), "cudaMalloc CPU-to-GPU dequeue position");
        check_cuda(cudaMalloc(&stats_device_, sizeof(ForwardDeviceQueueStats)),
                   "cudaMalloc CPU-to-GPU queue stats");
        check_cuda(cudaStreamCreateWithFlags(&publication_stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags CPU-to-GPU publication");
#else
        throw std::runtime_error("device-resident CPU-to-GPU queues require CUDA support");
#endif
        initialized_ = true;
    }

    void reset(uint64_t generation, void* stream) {
        producer_position_ = 0;
        mock_dequeue_position_.store(0, std::memory_order_relaxed);
        generation_ = generation;
        generated_.store(0);
        full_stalls_.store(0);
        mock_stats_ = {};
        store_release(&mock_published_head_, 0);
        for (uint64_t i = 0; i < capacity_; ++i) {
            tasks_host_[i] = {};
            if (publication_staging_) publication_staging_[i] = 0;
            store_release(&consumed_host_[i].sequence, i);
        }
#if RDMA_PROXY_HAVE_CUDA
        if (!mock_mode_) {
            check_cuda(cudaMemsetAsync(
                           published_head_device_, 0, sizeof(uint64_t), publication_stream_),
                       "cudaMemsetAsync CPU-to-GPU published head");
            check_cuda(cudaStreamSynchronize(publication_stream_),
                       "cudaStreamSynchronize CPU-to-GPU publication reset");
            check_cuda(cudaMemsetAsync(
                           dequeue_device_, 0, sizeof(uint64_t), reinterpret_cast<cudaStream_t>(stream)),
                       "cudaMemsetAsync CPU-to-GPU dequeue position");
            check_cuda(cudaMemsetAsync(
                           stats_device_, 0, sizeof(ForwardDeviceQueueStats), reinterpret_cast<cudaStream_t>(stream)),
                       "cudaMemsetAsync CPU-to-GPU queue stats");
        }
#else
        (void)stream;
#endif
    }

    void enqueue_batch(
        const std::vector<ForwardComputeTask>& tasks,
        std::chrono::steady_clock::time_point deadline,
        const std::atomic<bool>* abort_requested) {
        std::size_t task_index = 0;
        while (task_index < tasks.size()) {
            const std::size_t ring_index = static_cast<std::size_t>(producer_position_ % capacity_);
            const std::size_t batch_size = std::min<std::size_t>(
                tasks.size() - task_index, static_cast<std::size_t>(capacity_) - ring_index);

            for (std::size_t i = 0; i < batch_size; ++i) {
                const uint64_t position = producer_position_ + i;
                wait_for_slot(position, deadline, abort_requested);
                auto task = tasks[task_index + i];
                task.queue_id = id_;
                task.enqueue_timestamp_ns = steady_nanoseconds_now();
                tasks_host_[ring_index + i] = task;
                if (task.type == static_cast<uint32_t>(ForwardTaskType::kCompute)) {
                    generated_.fetch_add(1);
                }
            }

            if (mock_mode_) {
                store_release(&mock_published_head_, producer_position_ + batch_size);
                producer_position_ += batch_size;
                task_index += batch_size;
                continue;
            }
#if RDMA_PROXY_HAVE_CUDA
            // The descriptor copy and published-head copy share a stream. The
            // GPU can observe the advanced head only after every corresponding
            // descriptor byte has reached the device-resident task ring.
            check_cuda(cudaMemcpyAsync(
                           tasks_device_ + ring_index,
                           tasks_host_ + ring_index,
                           sizeof(ForwardComputeTask) * batch_size,
                           cudaMemcpyHostToDevice,
                           publication_stream_),
                       "cudaMemcpyAsync CPU-to-GPU task batch");
            const std::size_t publication_staging_index = ring_index + batch_size - 1;
            // This staging slot cannot be reused until the corresponding last
            // task completes, which is necessarily after this 8-byte DMA has
            // consumed the value.
            publication_staging_[publication_staging_index] = producer_position_ + batch_size;
            check_cuda(cudaMemcpyAsync(
                           published_head_device_,
                           publication_staging_ + publication_staging_index,
                           sizeof(uint64_t),
                           cudaMemcpyHostToDevice,
                           publication_stream_),
                       "cudaMemcpyAsync CPU-to-GPU publication batch");
#endif
            producer_position_ += batch_size;
            task_index += batch_size;
        }
    }

    bool try_claim_mock(ForwardComputeTask* task, uint64_t* position) {
        while (true) {
            uint64_t current = mock_dequeue_position_.load(std::memory_order_relaxed);
            if (current >= load_acquire(&mock_published_head_)) {
                return false;
            }
            if (mock_dequeue_position_.compare_exchange_weak(
                    current, current + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                *task = tasks_host_[current % capacity_];
                *position = current;
                return true;
            }
        }
    }

    void complete_mock(uint64_t position) {
        store_release(&consumed_host_[position % capacity_].sequence, position + capacity_);
    }

    ForwardDeviceQueueView device_view() const {
        ForwardDeviceQueueView view;
        view.published_head = published_head_device_;
        view.consumed = consumed_device_;
        view.tasks = tasks_device_;
        view.dequeue_position = static_cast<uint64_t*>(dequeue_device_);
        view.stats = static_cast<ForwardDeviceQueueStats*>(stats_device_);
        view.capacity = capacity_;
        view.queue_id = id_;
        view.generation = generation_;
        return view;
    }

    ForwardDeviceQueueStats device_stats() const {
        if (mock_mode_) {
            ForwardDeviceQueueStats result;
            result.poll_iterations = __atomic_load_n(&mock_stats_.poll_iterations, __ATOMIC_ACQUIRE);
            result.tasks_claimed = __atomic_load_n(&mock_stats_.tasks_claimed, __ATOMIC_ACQUIRE);
            result.tasks_completed = __atomic_load_n(&mock_stats_.tasks_completed, __ATOMIC_ACQUIRE);
            result.exit_tasks_consumed =
                __atomic_load_n(&mock_stats_.exit_tasks_consumed, __ATOMIC_ACQUIRE);
            result.stale_tasks = __atomic_load_n(&mock_stats_.stale_tasks, __ATOMIC_ACQUIRE);
            result.invalid_tasks = __atomic_load_n(&mock_stats_.invalid_tasks, __ATOMIC_ACQUIRE);
            return result;
        }
#if RDMA_PROXY_HAVE_CUDA
        ForwardDeviceQueueStats result{};
        check_cuda(cudaMemcpy(&result, stats_device_, sizeof(result), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H CPU-to-GPU queue stats");
        return result;
#else
        return {};
#endif
    }

    ForwardDeviceQueueStats& mock_stats() { return mock_stats_; }
    uint64_t generated() const { return generated_.load(); }
    uint64_t full_stalls() const { return full_stalls_.load(); }

private:
    void wait_for_slot(
        uint64_t position,
        std::chrono::steady_clock::time_point deadline,
        const std::atomic<bool>* abort_requested) {
        auto& consumed = consumed_host_[position % capacity_].sequence;
        bool recorded_stall = false;
        while (load_acquire(&consumed) != position) {
            if (!recorded_stall) {
                full_stalls_.fetch_add(1);
                recorded_stall = true;
            }
            if ((abort_requested && abort_requested->load(std::memory_order_acquire)) ||
                std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error(
                    "timed out waiting for CPU-to-GPU queue space queue=" + std::to_string(id_) +
                    " producer_position=" + std::to_string(position));
            }
            cpu_relax();
        }
    }

    template <typename T>
    void allocate_mock(T*& pointer, std::size_t bytes) {
        pointer = static_cast<T*>(::operator new(bytes, std::align_val_t(128)));
        std::memset(pointer, 0, bytes);
    }

#if RDMA_PROXY_HAVE_CUDA
    template <typename T>
    void allocate_host_staging(T*& host, std::size_t bytes) {
        void* raw = nullptr;
        check_cuda(cudaHostAlloc(&raw, bytes, cudaHostAllocPortable),
                   "cudaHostAlloc CPU-to-GPU staging ring");
        host = static_cast<T*>(raw);
        std::memset(host, 0, bytes);
    }

    template <typename T>
    void allocate_host_visible(T*& host, T*& device, std::size_t bytes) {
        void* raw = nullptr;
        const unsigned flags = cudaHostAllocMapped | cudaHostAllocPortable;
        check_cuda(cudaHostAlloc(&raw, bytes, flags),
                   "cudaHostAllocMapped GPU-to-CPU reuse ring");
        host = static_cast<T*>(raw);
        void* device_raw = nullptr;
        check_cuda(cudaHostGetDevicePointer(&device_raw, raw, 0),
                   "cudaHostGetDevicePointer GPU-to-CPU reuse ring");
        device = static_cast<T*>(device_raw);
        std::memset(host, 0, bytes);
    }
#endif

    template <typename T>
    void free_host_allocation(T*& host) noexcept {
        if (!host) return;
        if (mock_mode_) {
            ::operator delete(host, std::align_val_t(128));
        } else {
#if RDMA_PROXY_HAVE_CUDA
            const auto status = cudaFreeHost(host);
            if (status != cudaSuccess) {
                RDMA_PROXY_LOG_WARN("cudaFreeHost CPU/GPU queue allocation failed: ",
                                    cudaGetErrorString(status));
            }
#endif
        }
        host = nullptr;
    }

    void release() noexcept {
#if RDMA_PROXY_HAVE_CUDA
        if (!mock_mode_ && publication_stream_) {
            (void)cudaStreamSynchronize(publication_stream_);
            (void)cudaStreamDestroy(publication_stream_);
        }
        publication_stream_ = nullptr;
        if (!mock_mode_ && published_head_device_) (void)cudaFree(published_head_device_);
        if (!mock_mode_ && tasks_device_) (void)cudaFree(tasks_device_);
        if (!mock_mode_ && dequeue_device_) (void)cudaFree(dequeue_device_);
        if (!mock_mode_ && stats_device_) (void)cudaFree(stats_device_);
#endif
        dequeue_device_ = nullptr;
        stats_device_ = nullptr;
        free_host_allocation(publication_staging_);
        free_host_allocation(consumed_host_);
        free_host_allocation(tasks_host_);
        published_head_device_ = nullptr;
        consumed_device_ = nullptr;
        tasks_device_ = nullptr;
        initialized_ = false;
    }

    uint32_t id_{0};
    uint32_t capacity_{0};
    bool mock_mode_{false};
    bool initialized_{false};
    uint64_t generation_{0};
    uint64_t producer_position_{0};
    uint64_t mock_published_head_{0};
    std::atomic<uint64_t> mock_dequeue_position_{0};
    uint64_t* publication_staging_{nullptr};
    uint64_t* published_head_device_{nullptr};
    ForwardQueueSignal* consumed_host_{nullptr};
    ForwardQueueSignal* consumed_device_{nullptr};
    ForwardComputeTask* tasks_host_{nullptr};
    ForwardComputeTask* tasks_device_{nullptr};
    void* dequeue_device_{nullptr};
    void* stats_device_{nullptr};
#if RDMA_PROXY_HAVE_CUDA
    cudaStream_t publication_stream_{nullptr};
#endif
    std::atomic<uint64_t> generated_{0};
    std::atomic<uint64_t> full_stalls_{0};
    ForwardDeviceQueueStats mock_stats_{};
};

}  // namespace

std::vector<ForwardComputeTask> partition_forward_ready_region(const ForwardReadyRegion& region) {
    if (region.generation == 0 || region.valid_token_rows == 0 || region.matrix_n == 0 ||
        region.matrix_k == 0 || region.tile_m == 0 || region.tile_n == 0 ||
        region.a_base == 0 || region.b_base == 0 || region.d_base == 0) {
        throw std::runtime_error("invalid NVLink forwarding computation ready region");
    }
    if (region.valid_token_rows >
        std::numeric_limits<uint32_t>::max() - region.token_row_offset) {
        throw std::runtime_error("NVLink forwarding computation ready-region row offset overflows uint32");
    }
    const uint64_t row_tiles =
        (static_cast<uint64_t>(region.valid_token_rows) + region.tile_m - 1) / region.tile_m;
    const uint64_t column_tiles =
        (static_cast<uint64_t>(region.matrix_n) + region.tile_n - 1) / region.tile_n;
    if (row_tiles > std::numeric_limits<std::size_t>::max() / column_tiles) {
        throw std::runtime_error("NVLink forwarding computation task count overflows size_t");
    }
    std::vector<ForwardComputeTask> tasks;
    tasks.reserve(static_cast<std::size_t>(row_tiles * column_tiles));
    for (uint64_t row = 0; row < region.valid_token_rows; row += region.tile_m) {
        for (uint64_t column = 0; column < region.matrix_n; column += region.tile_n) {
            ForwardComputeTask task;
            task.type = static_cast<uint32_t>(ForwardTaskType::kCompute);
            task.dtype = region.dtype;
            task.generation = region.generation;
            task.receive_buffer_id = region.receive_buffer_id;
            task.output_buffer_id = region.output_buffer_id;
            task.token_row_offset = region.token_row_offset + static_cast<uint32_t>(row);
            task.valid_token_rows = static_cast<uint32_t>(
                std::min<uint64_t>(region.tile_m, region.valid_token_rows - row));
            task.output_column_offset = static_cast<uint32_t>(column);
            task.valid_output_columns = static_cast<uint32_t>(
                std::min<uint64_t>(region.tile_n, region.matrix_n - column));
            task.tile_m = region.tile_m;
            task.tile_n = region.tile_n;
            task.matrix_m = region.valid_token_rows;
            task.matrix_n = region.matrix_n;
            task.matrix_k = region.matrix_k;
            task.a_base = region.a_base;
            task.b_base = region.b_base;
            task.d_base = region.d_base;
            tasks.push_back(task);
        }
    }
    return tasks;
}

std::vector<uint32_t> partition_ctas_across_queues(uint32_t num_ctas, uint32_t num_queues) {
    if (num_ctas == 0 || num_queues == 0 || num_queues > num_ctas) {
        throw std::runtime_error("persistent CTA count must be >= the non-zero queue count");
    }
    std::vector<uint32_t> counts(num_queues, 0);
    for (uint32_t cta = 0; cta < num_ctas; ++cta) ++counts[cta % num_queues];
    return counts;
}

class ForwardComputation::Impl {
public:
    Impl(ProxyConfig config, CudaBuffers& buffers)
        : config_(std::move(config)), buffers_(buffers) {}

    ~Impl() { shutdown(); }

    void initialize() {
        if (initialized_ || !config_.nvlink_forward_computation_enabled) return;
#if RDMA_PROXY_HAVE_CUDA
        if (!config_.mock_mode) {
            check_cuda(cudaSetDevice(config_.cuda_device_id),
                       "cudaSetDevice before persistent computation setup");
        }
#endif
        const auto queue_count = static_cast<uint32_t>(config_.nvlink_forward_computation_num_queues);
        const auto capacity = static_cast<uint32_t>(config_.nvlink_forward_computation_queue_depth);
        queues_.reserve(queue_count);
        for (uint32_t q = 0; q < queue_count; ++q) {
            auto queue = std::make_unique<CpuToGpuQueue>(q, capacity, config_.mock_mode);
            queue->initialize();
            queues_.push_back(std::move(queue));
        }

        if (config_.mock_mode) {
            num_ctas_ = std::max<uint32_t>(queue_count + 1, queue_count * 2);
            ctas_per_queue_ = partition_ctas_across_queues(num_ctas_, queue_count);
            initialized_ = true;
            return;
        }
#if RDMA_PROXY_HAVE_CUDA
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, config_.cuda_device_id), "cudaGetDeviceProperties");
        if (properties.major < 9) {
            throw std::runtime_error(
                "NVLink forwarding computation requires Hopper (compute capability >= 9.0); found " +
                std::to_string(properties.major) + "." + std::to_string(properties.minor));
        }
        if (!properties.unifiedAddressing || !properties.canMapHostMemory) {
            throw std::runtime_error(
                "NVLink forwarding computation requires UVA and mapped pinned host memory");
        }
        num_ctas_ = static_cast<uint32_t>(properties.multiProcessorCount);
        if (queue_count >= num_ctas_) {
            throw std::runtime_error(
                "nvlink_forward_computation_num_queues must be smaller than the GPU SM count (queues=" +
                std::to_string(queue_count) + " SMs=" + std::to_string(num_ctas_) + ")");
        }
        ctas_per_queue_ = partition_ctas_across_queues(num_ctas_, queue_count);
        if (num_ctas_ % queue_count != 0) {
            RDMA_PROXY_LOG_WARN("persistent computation queues do not evenly divide SMs; logical CTA modulo "
                                "assignment differs by at most one CTA queue_count=",
                                queue_count, " SMs=", num_ctas_);
        }
        const std::size_t per_sm_shared = properties.sharedMemPerMultiprocessor;
        dynamic_shared_bytes_ = per_sm_shared / 2 + 256;
        if (dynamic_shared_bytes_ > static_cast<std::size_t>(properties.sharedMemPerBlockOptin)) {
            throw std::runtime_error(
                "GPU cannot reserve enough per-CTA shared memory to guarantee one persistent CTA per SM");
        }
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags persistent computation");
        check_cuda(cudaMalloc(&device_views_, sizeof(ForwardDeviceQueueView) * queue_count),
                   "cudaMalloc persistent queue views");
        check_cuda(cudaMalloc(&device_sm_ids_, sizeof(uint32_t) * num_ctas_),
                   "cudaMalloc persistent physical SM diagnostics");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&abort_device_),
                              sizeof(ForwardQueueSignal)),
                   "cudaMalloc persistent abort signal");
        check_cuda(cudaStreamCreateWithFlags(&control_stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags persistent control");
        initialized_ = true;
#else
        throw std::runtime_error("persistent computation requires a CUDA build or mock_mode=true");
#endif
    }

    void begin_iteration(uint64_t iteration) {
        if (!initialized_) initialize();
        if (active_.load()) throw std::runtime_error("persistent computation iteration is already active");
        active_iteration_ = iteration;
        const uint64_t generation = iteration + 1;
        if (generation == 0) throw std::runtime_error("persistent computation generation wrapped to zero");
        generation_ = generation;
        next_queue_ = 0;
        abort_requested_.store(false);
        generated_tasks_.store(0);
        enqueue_nanoseconds_.store(0);
        iteration_start_ = std::chrono::steady_clock::now();
        last_stats_ = {};
        last_stats_.ctas_per_queue = ctas_per_queue_;

        void* stream_ptr = nullptr;
#if RDMA_PROXY_HAVE_CUDA
        if (!config_.mock_mode) stream_ptr = reinterpret_cast<void*>(stream_);
#endif
        for (auto& queue : queues_) queue->reset(generation, stream_ptr);
        buffers_.clear_nvlink_computation_outputs(stream_ptr);

        active_.store(true, std::memory_order_release);
        if (config_.mock_mode) {
            mock_threads_.clear();
            mock_threads_.reserve(num_ctas_);
            for (uint32_t cta = 0; cta < num_ctas_; ++cta) {
                mock_threads_.emplace_back([this, cta]() { mock_consumer_loop(cta); });
            }
            return;
        }
#if RDMA_PROXY_HAVE_CUDA
        std::vector<ForwardDeviceQueueView> views;
        views.reserve(queues_.size());
        for (const auto& queue : queues_) views.push_back(queue->device_view());
        check_cuda(cudaMemsetAsync(abort_device_, 0, sizeof(ForwardQueueSignal), stream_),
                   "cudaMemsetAsync persistent abort signal");
        check_cuda(cudaMemcpyAsync(
                       device_views_,
                       views.data(),
                       sizeof(ForwardDeviceQueueView) * views.size(),
                       cudaMemcpyHostToDevice,
                       stream_),
                   "cudaMemcpyAsync persistent queue views");
        check_cuda(cudaMemsetAsync(device_sm_ids_, 0xff, sizeof(uint32_t) * num_ctas_, stream_),
                   "cudaMemsetAsync persistent physical SM diagnostics");
        launch_persistent_forward_computation_kernel(
            static_cast<const ForwardDeviceQueueView*>(device_views_),
            static_cast<uint32_t>(queues_.size()),
            num_ctas_,
            config_.dtype,
            config_.nvlink_forward_computation_load_only_enabled,
            abort_device_,
            static_cast<uint32_t*>(device_sm_ids_),
            dynamic_shared_bytes_,
            reinterpret_cast<void*>(stream_));
#endif
        RDMA_PROXY_LOG_INFO("launched persistent forwarding computation iteration=", iteration,
                            " generation=", generation,
                            " load_only=",
                            config_.nvlink_forward_computation_load_only_enabled ? "true" : "false",
                            " CTAs=", num_ctas_,
                            " queues=", queues_.size(),
                            " dynamic_shared_bytes=", dynamic_shared_bytes_);
    }

    std::size_t enqueue_ready_region(
        uint64_t iteration,
        int source_gpu,
        std::size_t peer_slot,
        std::size_t start_token,
        std::size_t num_tokens,
        std::size_t byte_offset) {
        if (!active_.load(std::memory_order_acquire) || iteration != active_iteration_) {
            throw std::runtime_error("forwarding computation notification does not match the active iteration");
        }
        if (num_tokens == 0) return 0;
        const auto row_bytes = config_.token_dimension * dtype_size(config_.dtype);
        if (row_bytes == 0 || byte_offset % row_bytes != 0) {
            throw std::runtime_error("forwarding computation notification byte offset is not row aligned");
        }
        const auto row_offset = byte_offset / row_bytes;
        const auto expected_row = peer_slot * config_.num_tokens + start_token;
        if (row_offset != expected_row) {
            throw std::runtime_error(
                "forwarding computation notification row metadata mismatch: byte_offset row=" +
                std::to_string(row_offset) + " peer/start row=" + std::to_string(expected_row));
        }
        const auto capacity_rows = config_.num_tokens * config_.peers.size();
        if (row_offset > capacity_rows || num_tokens > capacity_rows - row_offset) {
            throw std::runtime_error("forwarding computation notification exceeds receive-buffer capacity");
        }
        if (row_offset > std::numeric_limits<uint32_t>::max() ||
            num_tokens > std::numeric_limits<uint32_t>::max() ||
            num_tokens > std::numeric_limits<uint32_t>::max() - row_offset) {
            throw std::runtime_error("forwarding computation row metadata exceeds uint32 range");
        }
        const auto& receive = buffers_.nvlink_receive_buffer_for_source(source_gpu);
        const auto& output = buffers_.nvlink_computation_output_buffer_for_source(source_gpu);
        const auto& weight = buffers_.nvlink_computation_weight_buffer();
        ForwardReadyRegion region;
        region.generation = generation_;
        region.receive_buffer_id = static_cast<uint32_t>(source_gpu);
        region.output_buffer_id = static_cast<uint32_t>(source_gpu);
        region.token_row_offset = static_cast<uint32_t>(row_offset);
        region.valid_token_rows = static_cast<uint32_t>(num_tokens);
        region.matrix_n = static_cast<uint32_t>(config_.nvlink_forward_computation_output_dim);
        region.matrix_k = static_cast<uint32_t>(config_.token_dimension);
        region.tile_m = static_cast<uint32_t>(config_.nvlink_forward_computation_tile_m);
        region.tile_n = static_cast<uint32_t>(config_.nvlink_forward_computation_tile_n);
        region.dtype = static_cast<uint32_t>(config_.dtype);
        region.a_base = reinterpret_cast<uint64_t>(receive.recv.ptr);
        region.b_base = reinterpret_cast<uint64_t>(weight.ptr);
        region.d_base = reinterpret_cast<uint64_t>(output.output.ptr);
        auto tasks = partition_forward_ready_region(region);

        const auto enqueue_start = std::chrono::steady_clock::now();
        const auto deadline = enqueue_start + std::chrono::milliseconds(config_.completion_timeout_ms);
        std::vector<std::vector<ForwardComputeTask>> tasks_by_queue(queues_.size());
        for (auto& task : tasks) {
            tasks_by_queue[next_queue_].push_back(task);
            next_queue_ = (next_queue_ + 1) % queues_.size();
        }
        for (std::size_t q = 0; q < queues_.size(); ++q) {
            queues_[q]->enqueue_batch(tasks_by_queue[q], deadline, &abort_requested_);
        }
        const auto enqueue_end = std::chrono::steady_clock::now();
        enqueue_nanoseconds_.fetch_add(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(enqueue_end - enqueue_start).count()));
        generated_tasks_.fetch_add(tasks.size());
        if (config_.nvlink_forward_computation_log_enabled) {
            RDMA_PROXY_LOG_INFO("forwarding computation tasks generated iteration=", iteration,
                                " source_gpu=", source_gpu,
                                " peer_slot=", peer_slot,
                                " start_token=", start_token,
                                " rows=", num_tokens,
                                " tasks=", tasks.size());
        }
        return tasks.size();
    }

    void finish_iteration(uint64_t iteration) {
        if (!active_.load(std::memory_order_acquire) || iteration != active_iteration_) {
            throw std::runtime_error("cannot finish an inactive or mismatched persistent computation iteration");
        }
        try {
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(config_.completion_timeout_ms);
            for (std::size_t q = 0; q < queues_.size(); ++q) {
                std::vector<ForwardComputeTask> exits;
                exits.reserve(ctas_per_queue_[q]);
                for (uint32_t i = 0; i < ctas_per_queue_[q]; ++i) {
                    ForwardComputeTask exit;
                    exit.type = static_cast<uint32_t>(ForwardTaskType::kExit);
                    exit.dtype = static_cast<uint32_t>(config_.dtype);
                    exit.generation = generation_;
                    exits.push_back(exit);
                }
                queues_[q]->enqueue_batch(exits, deadline, &abort_requested_);
            }
            wait_for_kernel(deadline);
            collect_stats();
            if (last_stats_.exit_tasks_consumed != num_ctas_) {
                throw std::runtime_error(
                    "persistent computation exit count mismatch expected=" + std::to_string(num_ctas_) +
                    " observed=" + std::to_string(last_stats_.exit_tasks_consumed));
            }
            if (last_stats_.tasks_completed != generated_tasks_.load()) {
                throw std::runtime_error(
                    "persistent computation task completion mismatch generated=" +
                    std::to_string(generated_tasks_.load()) + " completed=" +
                    std::to_string(last_stats_.tasks_completed));
            }
            if (last_stats_.stale_tasks != 0 || last_stats_.invalid_tasks != 0) {
                throw std::runtime_error("persistent computation kernel reported stale or invalid tasks");
            }
        } catch (...) {
            // Keep the queue mappings alive until every GPU CTA has observed
            // the emergency abort and the kernel stream has terminated.
            abort_iteration();
            throw;
        }
        active_.store(false, std::memory_order_release);
        RDMA_PROXY_LOG_INFO("persistent forwarding computation complete iteration=", iteration,
                            " generated=", last_stats_.generated_tasks,
                            " claimed=", last_stats_.tasks_claimed,
                            " completed=", last_stats_.tasks_completed,
                            " exits=", last_stats_.exit_tasks_consumed,
                            " queue_full_stalls=", last_stats_.queue_full_stalls,
                            " poll_iterations=", last_stats_.poll_iterations,
                            " elapsed_ms=", last_stats_.iteration_seconds * 1.0e3,
                            " enqueue_ms=", last_stats_.enqueue_seconds * 1.0e3);
        if (config_.nvlink_forward_computation_log_enabled) {
            for (std::size_t q = 0; q < last_stats_.generated_by_queue.size(); ++q) {
                RDMA_PROXY_LOG_INFO("persistent forwarding computation queue iteration=", iteration,
                                    " queue=", q,
                                    " CTAs=", last_stats_.ctas_per_queue[q],
                                    " generated=", last_stats_.generated_by_queue[q],
                                    " completed=", last_stats_.completed_by_queue[q]);
            }
            for (std::size_t cta = 0; cta < last_stats_.physical_sm_ids.size(); ++cta) {
                RDMA_PROXY_LOG_INFO("persistent forwarding computation CTA placement iteration=", iteration,
                                    " logical_cta=", cta,
                                    " queue=", cta % queues_.size(),
                                    " physical_sm=", last_stats_.physical_sm_ids[cta]);
            }
        }
    }

    void abort_iteration() noexcept {
        if (!active_.load(std::memory_order_acquire)) return;
        signal_abort();
        join_mock_threads();
#if RDMA_PROXY_HAVE_CUDA
        if (!config_.mock_mode && stream_) {
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(config_.completion_timeout_ms);
            bool terminated = false;
            while (std::chrono::steady_clock::now() < deadline) {
                const auto status = cudaStreamQuery(stream_);
                if (status == cudaSuccess) {
                    terminated = true;
                    break;
                }
                if (status != cudaErrorNotReady) {
                    RDMA_PROXY_LOG_ERROR("persistent computation abort query failed: ",
                                         cudaGetErrorString(status));
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!terminated) {
                // Resource lifetime wins over a bounded teardown here: freeing
                // device queue storage while a CTA can still poll it is unsafe.
                const auto status = cudaStreamSynchronize(stream_);
                if (status != cudaSuccess) {
                    RDMA_PROXY_LOG_ERROR("persistent computation abort synchronization failed: ",
                                         cudaGetErrorString(status));
                }
            }
        }
#endif
        active_.store(false, std::memory_order_release);
    }

    void shutdown() noexcept {
        abort_iteration();
#if RDMA_PROXY_HAVE_CUDA
        if (!config_.mock_mode) {
            if (device_views_) (void)cudaFree(device_views_);
            if (device_sm_ids_) (void)cudaFree(device_sm_ids_);
            if (abort_device_) (void)cudaFree(abort_device_);
            if (control_stream_) (void)cudaStreamDestroy(control_stream_);
            if (stream_) (void)cudaStreamDestroy(stream_);
        }
#endif
        device_views_ = nullptr;
        device_sm_ids_ = nullptr;
        abort_device_ = nullptr;
#if RDMA_PROXY_HAVE_CUDA
        control_stream_ = nullptr;
        stream_ = nullptr;
#endif
        queues_.clear();
        initialized_ = false;
    }

    bool wait_until_completed(uint64_t count, std::chrono::milliseconds timeout) const {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            uint64_t completed = 0;
            if (config_.mock_mode) {
                for (const auto& queue : queues_) completed += queue->device_stats().tasks_completed;
            } else {
#if RDMA_PROXY_HAVE_CUDA
                for (const auto& queue : queues_) completed += queue->device_stats().tasks_completed;
#endif
            }
            if (completed >= count) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }

    ForwardComputationStats stats() const {
        if (!active_.load(std::memory_order_acquire)) return last_stats_;
        ForwardComputationStats result;
        result.generated_tasks = generated_tasks_.load();
        result.ctas_per_queue = ctas_per_queue_;
        for (const auto& queue : queues_) {
            const auto qstats = queue->device_stats();
            result.generated_by_queue.push_back(queue->generated());
            result.completed_by_queue.push_back(qstats.tasks_completed);
            result.queue_full_stalls += queue->full_stalls();
            result.poll_iterations += qstats.poll_iterations;
            result.tasks_claimed += qstats.tasks_claimed;
            result.tasks_completed += qstats.tasks_completed;
            result.exit_tasks_consumed += qstats.exit_tasks_consumed;
            result.stale_tasks += qstats.stale_tasks;
            result.invalid_tasks += qstats.invalid_tasks;
        }
        return result;
    }

    bool iteration_active() const { return active_.load(std::memory_order_acquire); }

private:
    void execute_mock_task(const ForwardComputeTask& task) {
        const auto dtype = static_cast<DataType>(task.dtype);
        const auto* a = reinterpret_cast<const uint16_t*>(task.a_base);
        const auto* b = reinterpret_cast<const uint16_t*>(task.b_base);
        auto* d = reinterpret_cast<uint16_t*>(task.d_base);
        for (uint32_t local_row = 0; local_row < task.valid_token_rows; ++local_row) {
            const uint64_t row = static_cast<uint64_t>(task.token_row_offset) + local_row;
            for (uint32_t local_col = 0; local_col < task.valid_output_columns; ++local_col) {
                const uint64_t column = static_cast<uint64_t>(task.output_column_offset) + local_col;
                float sum = 0.0F;
                for (uint32_t k = 0; k < task.matrix_k; ++k) {
                    sum += decode_value(a[row * task.matrix_k + k], dtype) *
                           decode_value(b[static_cast<uint64_t>(k) * task.matrix_n + column], dtype);
                }
                d[row * task.matrix_n + column] = encode_value(sum, dtype);
            }
        }
    }

    void mock_consumer_loop(uint32_t cta) {
        const auto queue_index = cta % queues_.size();
        auto& queue = *queues_[queue_index];
        uint64_t empty_polls = 0;
        while (!abort_requested_.load(std::memory_order_acquire)) {
            ForwardComputeTask task;
            uint64_t position = 0;
            if (!queue.try_claim_mock(&task, &position)) {
                ++empty_polls;
                if ((empty_polls & 0xffU) == 0) std::this_thread::yield();
                continue;
            }
            auto& stats = queue.mock_stats();
            if (task.generation != generation_) {
                __atomic_fetch_add(&stats.stale_tasks, 1ULL, __ATOMIC_RELAXED);
                queue.complete_mock(position);
                continue;
            }
            if (task.type == static_cast<uint32_t>(ForwardTaskType::kExit)) {
                __atomic_fetch_add(&stats.exit_tasks_consumed, 1ULL, __ATOMIC_RELAXED);
                queue.complete_mock(position);
                __atomic_fetch_add(&stats.poll_iterations, empty_polls, __ATOMIC_RELAXED);
                return;
            }
            if (task.type != static_cast<uint32_t>(ForwardTaskType::kCompute)) {
                __atomic_fetch_add(&stats.invalid_tasks, 1ULL, __ATOMIC_RELAXED);
                queue.complete_mock(position);
                continue;
            }
            __atomic_fetch_add(&stats.tasks_claimed, 1ULL, __ATOMIC_RELAXED);
            if (!config_.nvlink_forward_computation_load_only_enabled) {
                execute_mock_task(task);
            }
            __atomic_fetch_add(&stats.tasks_completed, 1ULL, __ATOMIC_RELEASE);
            queue.complete_mock(position);
        }
        __atomic_fetch_add(&queue.mock_stats().poll_iterations, empty_polls, __ATOMIC_RELAXED);
    }

    void signal_abort() noexcept {
        abort_requested_.store(true, std::memory_order_release);
#if RDMA_PROXY_HAVE_CUDA
        if (!config_.mock_mode && abort_device_ && control_stream_) {
            // The abort flag is device-resident because it is polled in the
            // kernel's hot loop. A dedicated nonblocking stream lets the copy
            // engine update it even while the persistent kernel occupies SMs.
            auto status = cudaMemsetAsync(abort_device_, 1, sizeof(uint64_t), control_stream_);
            if (status == cudaSuccess) status = cudaStreamSynchronize(control_stream_);
            if (status != cudaSuccess) {
                RDMA_PROXY_LOG_ERROR("persistent computation abort publication failed: ",
                                     cudaGetErrorString(status));
            }
        }
#endif
    }

    void join_mock_threads() noexcept {
        for (auto& thread : mock_threads_) {
            if (thread.joinable()) thread.join();
        }
        mock_threads_.clear();
    }

    void wait_for_kernel(std::chrono::steady_clock::time_point deadline) {
        if (config_.mock_mode) {
            while (std::chrono::steady_clock::now() < deadline) {
                bool any_joinable = false;
                for (const auto& thread : mock_threads_) any_joinable = any_joinable || thread.joinable();
                if (!any_joinable) break;
                uint64_t exits = 0;
                for (const auto& queue : queues_) exits += queue->device_stats().exit_tasks_consumed;
                if (exits == num_ctas_) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            join_mock_threads();
            uint64_t exits = 0;
            for (const auto& queue : queues_) exits += queue->device_stats().exit_tasks_consumed;
            if (exits != num_ctas_) throw std::runtime_error("timed out waiting for mock persistent CTAs to exit");
            return;
        }
#if RDMA_PROXY_HAVE_CUDA
        while (std::chrono::steady_clock::now() < deadline) {
            const auto status = cudaStreamQuery(stream_);
            if (status == cudaSuccess) return;
            if (status != cudaErrorNotReady) check_cuda(status, "persistent computation kernel termination");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error("timed out waiting for persistent computation kernel termination");
#endif
    }

    void collect_stats() {
        last_stats_ = {};
        last_stats_.generated_tasks = generated_tasks_.load();
        last_stats_.enqueue_seconds = static_cast<double>(enqueue_nanoseconds_.load()) / 1.0e9;
        last_stats_.iteration_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_start_).count();
        last_stats_.ctas_per_queue = ctas_per_queue_;
        last_stats_.generated_by_queue.reserve(queues_.size());
        last_stats_.completed_by_queue.reserve(queues_.size());
        for (const auto& queue : queues_) {
            const auto qstats = queue->device_stats();
            last_stats_.generated_by_queue.push_back(queue->generated());
            last_stats_.completed_by_queue.push_back(qstats.tasks_completed);
            last_stats_.queue_full_stalls += queue->full_stalls();
            last_stats_.poll_iterations += qstats.poll_iterations;
            last_stats_.tasks_claimed += qstats.tasks_claimed;
            last_stats_.tasks_completed += qstats.tasks_completed;
            last_stats_.exit_tasks_consumed += qstats.exit_tasks_consumed;
            last_stats_.stale_tasks += qstats.stale_tasks;
            last_stats_.invalid_tasks += qstats.invalid_tasks;
        }
#if RDMA_PROXY_HAVE_CUDA
        if (!config_.mock_mode) {
            last_stats_.physical_sm_ids.resize(num_ctas_);
            check_cuda(cudaMemcpy(
                           last_stats_.physical_sm_ids.data(),
                           device_sm_ids_,
                           sizeof(uint32_t) * num_ctas_,
                           cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H persistent physical SM diagnostics");
        }
#endif
    }

    ProxyConfig config_;
    CudaBuffers& buffers_;
    std::vector<std::unique_ptr<CpuToGpuQueue>> queues_;
    std::vector<std::thread> mock_threads_;
    std::vector<uint32_t> ctas_per_queue_;
    uint32_t num_ctas_{0};
    std::size_t dynamic_shared_bytes_{0};
    std::size_t next_queue_{0};
    uint64_t generation_{0};
    uint64_t active_iteration_{0};
    std::atomic<bool> active_{false};
    std::atomic<bool> abort_requested_{false};
    bool initialized_{false};
    std::atomic<uint64_t> generated_tasks_{0};
    std::atomic<uint64_t> enqueue_nanoseconds_{0};
    std::chrono::steady_clock::time_point iteration_start_{};
    ForwardComputationStats last_stats_;
    void* device_views_{nullptr};
    void* device_sm_ids_{nullptr};
    ForwardQueueSignal* abort_device_{nullptr};
#if RDMA_PROXY_HAVE_CUDA
    cudaStream_t control_stream_{nullptr};
    cudaStream_t stream_{nullptr};
#endif
};

ForwardComputation::ForwardComputation(ProxyConfig config, CudaBuffers& buffers)
    : impl_(new Impl(std::move(config), buffers)) {}

ForwardComputation::~ForwardComputation() = default;

void ForwardComputation::initialize() { impl_->initialize(); }
void ForwardComputation::begin_iteration(uint64_t iteration) { impl_->begin_iteration(iteration); }
std::size_t ForwardComputation::enqueue_ready_region(
    uint64_t iteration,
    int source_gpu,
    std::size_t peer_slot,
    std::size_t start_token,
    std::size_t num_tokens,
    std::size_t byte_offset) {
    return impl_->enqueue_ready_region(
        iteration, source_gpu, peer_slot, start_token, num_tokens, byte_offset);
}
void ForwardComputation::finish_iteration(uint64_t iteration) { impl_->finish_iteration(iteration); }
void ForwardComputation::abort_iteration() noexcept { impl_->abort_iteration(); }
void ForwardComputation::shutdown() noexcept { impl_->shutdown(); }
bool ForwardComputation::wait_until_completed(uint64_t count, std::chrono::milliseconds timeout) const {
    return impl_->wait_until_completed(count, timeout);
}
ForwardComputationStats ForwardComputation::stats() const { return impl_->stats(); }
bool ForwardComputation::iteration_active() const { return impl_->iteration_active(); }

}  // namespace rdma_proxy
