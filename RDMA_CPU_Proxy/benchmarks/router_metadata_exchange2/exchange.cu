#include "exchange.cuh"
#include "metadata_view.hpp"
#include "two_node_layout.hpp"
#include <cuda/atomic>

// One CTA per destination; no intermediate packing allocation. Self only
// writes its small header: its D2H payload is read directly from CuTe output.
__global__ void exchange_kernel(PeerArenas peers, int source, int capacity,
                                const int32_t* offsets, const int32_t* values) {
    const int dest = blockIdx.x;
    const int begin = offsets[dest * rme2::epg];
    const int end = offsets[(dest + 1) * rme2::epg];
    int32_t* header = peers.ptr[dest] + source * rme2::header_words;
    if (begin < 0 || end < begin || end > capacity) {
        if (threadIdx.x == 0) header[0] = -1;
        __threadfence_system();
        return;
    }
    if (threadIdx.x == 0) header[0] = begin;
    if (threadIdx.x <= rme2::epg)
        header[1 + threadIdx.x] = offsets[dest * rme2::epg + threadIdx.x] - begin;
    if (dest != source) {
        int32_t* dst = peers.ptr[dest] + rme2::gpus * rme2::header_words
                       + size_t(source) * capacity;
        for (int i = threadIdx.x; i < end - begin; i += blockDim.x)
            dst[i] = values[begin + i];
    }
    // Every writer fences its own writes before the stream completion event.
    __threadfence_system();
}

void launch_exchange(PeerArenas peers, int source, int capacity,
                     const int32_t* offsets, const int32_t* values, cudaStream_t stream) {
    exchange_kernel<<<rme2::gpus, 256, 0, stream>>>(peers, source, capacity, offsets, values);
}

// One CTA per global destination. Each local relay owns two banks of eight
// original-source records. Self payloads are copied too, so all RDMA sources
// belong to the relay's registered allocation (never an imported IPC pointer).
__global__ void exchange_two_kernel(PeerArenas peers, int source, int node, int capacity,
                                    uint64_t iteration, const int32_t* offsets,
                                    const int32_t* values, const int32_t* node_offsets,
                                    const int32_t* x3, const uint8_t* x4, bool direct_x3_x4_output,
                                    bool receiver_kernel, size_t flags_offset) {
    const int dest = blockIdx.x, relay = dest % 8, slot = (dest / 8) * 8 + source;
    int32_t* arena = peers.ptr[relay];
    int32_t* record = arena + rme2::two::directory_words +
                      size_t(slot) * (size_t(capacity) + rme2::two::header_words);
    int32_t* directory = arena + slot * rme2::two::header_words;
    const int begin = offsets[dest * 8], end = offsets[(dest + 1) * 8];
    if (threadIdx.x < rme2::two::header_words) {
        const int t = threadIdx.x;
        int32_t value = 0;
        if (t == 0) value = rme2::two::magic;
        else if (t == 1) value = rme2::two::version;
        else if (t == 2) value = int32_t(uint32_t(iteration));
        else if (t == 3) value = int32_t(uint32_t(iteration >> 32));
        else if (t == 4) value = node * 8 + source;
        else if (t == 5) value = dest;
        else if (t == 6) value = begin;
        else value = offsets[dest * 8 + t - 7] - begin;
        if (begin < 0 || end < begin || end > capacity) value = -1;
        directory[t] = record[t] = value;
    }
    if (begin >= 0 && end >= begin && end <= capacity)
        for (int i = threadIdx.x; i < end - begin; i += blockDim.x)
            record[rme2::two::header_words + i] = values[begin + i];
    // Only this source's same-index remote CTA prepares the x3/x4 header.
    // Direct-output mode has already written the payload in metadata kernel 4.
    // Otherwise copy it here. The NIC always reads the owner's registered arena.
    if (dest == (1 - node) * 8 + source) {
        const int tokens = capacity / 8, remote_node = 1 - node;
        const int first = node_offsets[remote_node], last = node_offsets[remote_node + 1];
        const bool valid = first >= 0 && last >= first && last <= 2 * tokens && last - first <= tokens;
        const int count = valid ? last - first : -1;
        int32_t* route_record = arena + rme2::two::directory_words +
            size_t(rme2::two::sources) * (size_t(capacity) + rme2::two::header_words);
        int32_t* route_header = arena + rme2::two::sources * rme2::two::header_words;
        if (threadIdx.x < rme2::two::header_words) {
            const int t = threadIdx.x;
            int32_t value = 0;
            if (t == 0) value = rme2::two::x3_x4_magic;
            else if (t == 1) value = rme2::two::version;
            else if (t == 2) value = int32_t(uint32_t(iteration));
            else if (t == 3) value = int32_t(uint32_t(iteration >> 32));
            else if (t == 4) value = node * 8 + source;
            else if (t == 5) value = dest;
            else if (t == 6) value = count;
            route_header[t] = route_record[t] = value;
        }
        if (valid && !direct_x3_x4_output) {
            int32_t* out_x3 = route_record + rme2::two::header_words;
            auto* out_x4 = reinterpret_cast<uint8_t*>(out_x3 + count);
            for (int i = threadIdx.x; i < count; i += blockDim.x) {
                out_x3[i] = x3[first + i];
                out_x4[i] = x4[first + i];
            }
        }
    }
    __threadfence_system();
    if (receiver_kernel) {
        // Every writer fenced its payload, then joins before the publishing
        // thread releases the flag. Headers alone are never readiness signals.
        __syncthreads();
        if (threadIdx.x == 0) {
            auto* flags = reinterpret_cast<rme2::receiver::Flag*>(
                reinterpret_cast<char*>(arena) + flags_offset);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(flags[slot].epoch)
                .store(iteration, cuda::memory_order_release);
            if (dest == (1 - node) * 8 + source)
                cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(flags[rme2::two::sources].epoch)
                    .store(iteration, cuda::memory_order_release);
        }
    }
}
void launch_exchange_two(PeerArenas peers, int source, int node, int capacity,
                         uint64_t iteration, const int32_t* offsets,
                         const int32_t* values, const int32_t* node_offsets,
                         const int32_t* x3, const uint8_t* x4, bool direct_x3_x4_output,
                         bool receiver_kernel,
                         cudaStream_t stream) {
    exchange_two_kernel<<<16, 256, 0, stream>>>(peers, source, node, capacity, iteration,
                                             offsets, values, node_offsets, x3, x4, direct_x3_x4_output,
                                             receiver_kernel, rme2::receiver::flags_offset(capacity));
}

__global__ void receiver_kernel(int32_t* inbox, int32_t* host, int rank, int node,
                                int capacity, uint64_t iteration, uint64_t timeout_cycles,
                                size_t flags_offset, rme2::receiver::Result* results) {
    namespace two = rme2::two;
    const int slot = int(blockIdx.x);
    const bool route_header = slot == two::sources;
    const bool local_record = !route_header && slot / 8 == node;
    auto* flags = reinterpret_cast<rme2::receiver::Flag*>(
        reinterpret_cast<char*>(inbox) + flags_offset);
    const size_t offset = two::directory_words +
        size_t(slot) * (size_t(capacity) + two::header_words);
    // Read the record header, published together with its directory duplicate.
    const int32_t* record = inbox + offset;
    __shared__ int count, error;
    if (threadIdx.x == 0) {
        error = 0;
        count = 0;
        const auto started = clock64();
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ready(flags[slot].epoch);
        while (ready.load(cuda::memory_order_acquire) != iteration) {
            if (clock64() - started >= timeout_cycles) {
                error = rme2::receiver::timeout_error;
                break;
            }
            __nanosleep(64);
        }
        if (!error) {
            const uint64_t epoch = uint64_t(uint32_t(record[2])) |
                (uint64_t(uint32_t(record[3])) << 32);
            if (record[1] != two::version || epoch != iteration)
                error = rme2::receiver::record_error;
            if (route_header) {
                if (record[0] != two::x3_x4_magic || record[4] != node * 8 + rank ||
                    record[5] != (1 - node) * 8 + rank || record[6] < 0 || record[6] > capacity / 8)
                    error = rme2::receiver::record_error;
                for (int i = 7; i < two::header_words; ++i)
                    if (record[i] != 0) error = rme2::receiver::record_error;
            } else {
                count = record[15];
                if (record[0] != two::magic || record[4] != node * 8 + slot % 8 ||
                    record[5] != (slot / 8) * 8 + rank || record[6] < 0 || record[6] > capacity ||
                    record[7] != 0 || count < 0 || count > capacity - record[6])
                    error = rme2::receiver::record_error;
                for (int i = 7; i < 15; ++i)
                    if (record[i] > record[i + 1]) error = rme2::receiver::record_error;
            }
        }
    }
    // Propagate the acquiring thread's visibility to all copying threads.
    __syncthreads();
    if (!error) {
        if (threadIdx.x < two::header_words)
            host[slot * two::header_words + threadIdx.x] = record[threadIdx.x];
        // Include self. The other bank and the host x3/x4 record belong to
        // incoming RDMA; only their OUTGOING directory headers are read back.
        if (local_record)
            for (int i = threadIdx.x; i < two::header_words + count; i += blockDim.x)
                host[offset + i] = record[i];
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        results[slot].error = error;
        results[slot].epoch = iteration;
    }
}

void launch_receiver(int32_t* inbox, int32_t* mapped_host, int rank, int node,
                     int capacity, uint64_t iteration, uint64_t timeout_cycles,
                     rme2::receiver::Result* results, cudaStream_t stream) {
    receiver_kernel<<<rme2::receiver::records, 256, 0, stream>>>(inbox, mapped_host, rank, node, capacity,
        iteration, timeout_cycles, rme2::receiver::flags_offset(capacity), results);
}
