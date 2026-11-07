// Remote-only, eight peer-accessible GPUs; exercises the actual exchange and
// receiver kernels without Python, CuTe, IPC setup, or an RDMA fabric.
#include "exchange.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
__global__ void delay(uint64_t cycles) {
    const auto begin = clock64();
    while (clock64() - begin < cycles) __nanosleep(256);
}
// Stand in for work queued after the receiver: read its host output on the
// work stream without any intervening CPU synchronization. A delayed producer
// must not let this consumer observe stale headers or partially copied payloads.
__global__ void consume_host(const int32_t* arena, const int32_t* host, int node,
                              int capacity, uint64_t epoch,
                              const rme2::receiver::Result* results, int32_t* status) {
    namespace two = rme2::two;
    __shared__ int bad;
    if (threadIdx.x == 0) bad = 0;
    __syncthreads();
    for (int slot = 0; slot < rme2::receiver::records; ++slot) {
        const size_t offset = two::directory_words + size_t(slot) * (capacity + two::header_words);
        if (threadIdx.x == 0 && (results[slot].epoch != epoch || results[slot].error))
            atomicOr(&bad, 1);
        if (threadIdx.x < two::header_words &&
            host[slot * two::header_words + threadIdx.x] != arena[offset + threadIdx.x])
            atomicOr(&bad, 1);
        if (slot < two::sources && slot / 8 == node) {
            const int count = arena[offset + 15];
            for (int i = threadIdx.x; i < two::header_words + count; i += blockDim.x)
                if (host[offset + i] != arena[offset + i]) atomicOr(&bad, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) *status = bad;
}
struct Device {
    int32_t *arena{}, *host{}, *mapped{}, *offsets{}, *values{}, *nodes{};
    int32_t *consumer_status{}, *mapped_consumer_status{};
    rme2::receiver::Result *results{}, *mapped_results{};
    cudaStream_t send{}, receive{};
    cudaEvent_t exchange_start{}, sent{}, received{}, consumed{};
    uint64_t cycles_per_ms{};
    std::vector<int32_t> offsets_cpu, values_cpu;
};
constexpr int sentinel = -777777;

void run(int tokens, bool overlap_exchange) {
    using namespace rme2;
    const int capacity = tokens * 8;
    const size_t words = two::words(capacity);
    std::array<Device, 8> gpu{};
    PeerArenas peers{};
    for (int g = 0; g < 8; ++g) {
        check(cudaSetDevice(g));
        auto& d = gpu[g];
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, g));
        d.cycles_per_ms = prop.clockRate;
        check(cudaMalloc(reinterpret_cast<void**>(&d.arena), receiver::allocation_bytes(capacity)));
        check(cudaMemset(d.arena, 0, receiver::allocation_bytes(capacity)));
        peers.ptr[g] = d.arena;
        check(cudaHostAlloc(reinterpret_cast<void**>(&d.host), words * 4, cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d.mapped), d.host, 0));
        check(cudaHostAlloc(reinterpret_cast<void**>(&d.results), receiver::records * sizeof(receiver::Result), cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d.mapped_results), d.results, 0));
        check(cudaHostAlloc(reinterpret_cast<void**>(&d.consumer_status), 4, cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d.mapped_consumer_status), d.consumer_status, 0));
        check(cudaMalloc(reinterpret_cast<void**>(&d.offsets), 129 * 4));
        check(cudaMalloc(reinterpret_cast<void**>(&d.values), capacity * 4));
        check(cudaMalloc(reinterpret_cast<void**>(&d.nodes), 3 * 4));
        check(cudaMemset(d.nodes, 0, 3 * 4)); // x3/x4 slices empty in this transport test
        check(cudaStreamCreateWithFlags(&d.send, cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&d.receive, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&d.exchange_start, cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&d.sent, cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&d.received, cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&d.consumed, cudaEventDisableTiming));
        check(cudaDeviceSynchronize());
    }
    uint64_t epoch = uint64_t{1} << 32; // Exercise both halves of the wire epoch.
    for (int node = 0; node < 2; ++node) {
        // Reuse every allocation/flag. Include empty, balanced, concentrated,
        // shrinking and growing records, both banks and partial CUDA blocks.
        for (int pattern = 0; pattern < 5; ++pattern) {
            ++epoch;
            for (int source = 0; source < 8; ++source) {
                check(cudaSetDevice(source));
                auto& d = gpu[source];
                std::fill(d.host, d.host + words, sentinel);
                *d.consumer_status = sentinel;
                for (int s = 0; s < receiver::records; ++s) d.results[s] = {};
                d.offsets_cpu.assign(129, 0);
                d.values_cpu.resize(capacity);
                const int routes = pattern == 0 ? 0 : pattern == 4 ? capacity / 3 : capacity;
                for (int i = 0; i < routes; ++i) {
                    const int e = pattern == 2 ? ((source + 1) % 8 + node * 8) * 8 :
                                  pattern == 3 ? ((source + 3) % 8 + (1 - node) * 8) * 8 : i % 128;
                    ++d.offsets_cpu[e + 1];
                }
                for (int e = 0; e < 128; ++e) d.offsets_cpu[e + 1] += d.offsets_cpu[e];
                for (int i = 0; i < capacity; ++i)
                    d.values_cpu[i] = int(epoch & 0xffff) * 10000000 + source * 1000000 + i;
                check(cudaMemcpy(d.offsets, d.offsets_cpu.data(), 129 * 4, cudaMemcpyHostToDevice));
                check(cudaMemcpy(d.values, d.values_cpu.data(), capacity * 4, cudaMemcpyHostToDevice));
            }
            // Delay one producer. Early receivers must wait for its current
            // epoch rather than reading last iteration's payload or header.
            for (int source = 0; source < 8; ++source) {
                check(cudaSetDevice(source));
                auto& d = gpu[source];
                // In overlap mode, GPU 7's receiver may proceed while its own
                // outbound work is delayed; all incoming records still need
                // this iteration's peer publications. No global start barrier.
                check(cudaEventRecord(d.exchange_start, d.send));
                if (source == 7) delay<<<1, 1, 0, d.send>>>(d.cycles_per_ms * 5);
                launch_exchange_two(peers, source, node, capacity, epoch, d.offsets, d.values,
                                    d.nodes, nullptr, nullptr, false, true, d.send);
                check(cudaGetLastError());
                check(cudaEventRecord(d.sent, d.send));
                check(cudaStreamWaitEvent(d.receive, overlap_exchange ? d.exchange_start : d.sent, 0));
                launch_receiver(d.arena, d.mapped, source, node, capacity, epoch,
                                d.cycles_per_ms * 10000, d.mapped_results, d.receive);
                check(cudaGetLastError());
                check(cudaEventRecord(d.received, d.receive));
                check(cudaStreamWaitEvent(d.send, d.received, 0));
                consume_host<<<1, 256, 0, d.send>>>(d.arena, d.mapped, node, capacity, epoch,
                                                   d.mapped_results, d.mapped_consumer_status);
                check(cudaGetLastError());
                check(cudaEventRecord(d.consumed, d.send));
            }
            for (int dest = 0; dest < 8; ++dest) {
                check(cudaSetDevice(dest));
                auto& d = gpu[dest];
                check(cudaEventSynchronize(d.received));
                // A receiver can now finish before its local exchange. Both
                // streams must drain before inputs/arenas are reused or freed.
                check(cudaEventSynchronize(d.sent));
                check(cudaEventSynchronize(d.consumed));
                require(*d.consumer_status == 0, "downstream consumer ran before all host stores completed");
                std::vector<int32_t> expected(words, sentinel);
                for (int slot = 0; slot < receiver::records; ++slot)
                    require(d.results[slot].epoch == epoch && !d.results[slot].error,
                            "receiver completion mismatch");
                for (int slot = 0; slot < two::sources; ++slot) {
                    const int source = slot % 8;
                    const auto& input = gpu[source];
                    const int global_dest = (slot / 8) * 8 + dest;
                    const int first = input.offsets_cpu[global_dest * 8];
                    const int count = input.offsets_cpu[(global_dest + 1) * 8] - first;
                    auto* out = expected.data() + slot * two::header_words;
                    out[0] = two::magic; out[1] = two::version;
                    out[2] = int32_t(uint32_t(epoch)); out[3] = int32_t(uint32_t(epoch >> 32));
                    out[4] = node * 8 + source; out[5] = global_dest; out[6] = first;
                    for (int e = 0; e <= 8; ++e)
                        out[7 + e] = input.offsets_cpu[global_dest * 8 + e] - first;
                    if (slot / 8 == node) {
                        auto* record = expected.data() + two::record_offset(slot, capacity);
                        std::copy_n(out, two::header_words, record);
                        std::copy_n(input.values_cpu.data() + first, count, record + two::header_words);
                    }
                }
                auto* route_header = expected.data() + two::sources * two::header_words;
                std::fill_n(route_header, two::header_words, 0);
                route_header[0] = two::x3_x4_magic; route_header[1] = two::version;
                route_header[2] = int32_t(uint32_t(epoch)); route_header[3] = int32_t(uint32_t(epoch >> 32));
                route_header[4] = node * 8 + dest; route_header[5] = (1 - node) * 8 + dest;
                // Includes self, the entire directory, untouched incoming RDMA
                // bank/x3/x4 receive record, and every unused capacity tail.
                require(std::equal(expected.begin(), expected.end(), d.host), "payload/guard mismatch");
            }
        }
    }
    check(cudaSetDevice(0));
    auto& d = gpu[0];
    const auto bad_slot = two::record_offset(8 + 1, capacity);
    const int32_t invalid_count = capacity + 1;
    check(cudaMemcpy(d.arena + bad_slot + 15, &invalid_count, 4, cudaMemcpyHostToDevice));
    std::fill(d.host, d.host + words, sentinel);
    launch_receiver(d.arena, d.mapped, 0, 1, capacity, epoch, d.cycles_per_ms * 1000,
                    d.mapped_results, d.receive);
    check(cudaStreamSynchronize(d.receive));
    require(d.results[9].error == receiver::record_error, "invalid count was not rejected");
    require(d.host[bad_slot] == sentinel && d.host[9 * two::header_words] == sentinel, "invalid record overwrote host memory");
    std::fill(d.host, d.host + words, sentinel);
    launch_receiver(d.arena, d.mapped, 0, 1, capacity, epoch + 1, d.cycles_per_ms * 5,
                    d.mapped_results, d.receive);
    check(cudaStreamSynchronize(d.receive));
    for (int s = 0; s < receiver::records; ++s)
        require(d.results[s].epoch == epoch + 1 && d.results[s].error == receiver::timeout_error,
                "missing producer/stale epoch did not time out");
    require(std::all_of(d.host, d.host + words, [](int x) { return x == sentinel; }),
            "stale epoch copied data");
    for (int g = 0; g < 8; ++g) {
        check(cudaSetDevice(g));
        auto& item = gpu[g];
        check(cudaStreamDestroy(item.send)); check(cudaStreamDestroy(item.receive));
        check(cudaEventDestroy(item.sent)); check(cudaEventDestroy(item.received));
        check(cudaEventDestroy(item.exchange_start)); check(cudaEventDestroy(item.consumed));
        check(cudaFree(item.arena)); check(cudaFree(item.offsets));
        check(cudaFree(item.values)); check(cudaFree(item.nodes));
        check(cudaFreeHost(item.host)); check(cudaFreeHost(item.results));
        check(cudaFreeHost(item.consumer_status));
    }
}
} // namespace
int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8) return 77;
        for (int source = 0; source < 8; ++source) {
            check(cudaSetDevice(source));
            cudaDeviceProp prop{};
            check(cudaGetDeviceProperties(&prop, source));
            if (prop.major < 9 || !prop.canMapHostMemory || !prop.concurrentKernels) return 77;
            for (int dest = 0; dest < 8; ++dest) if (source != dest) {
                int supported = 0;
                check(cudaDeviceGetP2PAttribute(&supported, cudaDevP2PAttrNativeAtomicSupported, source, dest));
                if (!supported) return 77;
                const auto status = cudaDeviceEnablePeerAccess(dest, 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                else check(status);
            }
        }
        for (bool overlap : {false, true}) {
            for (int tokens : {1, 33, 257, 20480}) run(tokens, overlap);
            std::cout << "receiver CUDA: overlap_exchange=" << overlap
                      << "; publication, self/directory readback, downstream ordering, guards and timeout passed\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
