#include "router_routing.hpp"

#include "logging.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime.h>
#include "expert_token_idx_standalone.hpp"
#endif

namespace rdma_proxy {
namespace {

constexpr std::size_t kKernelChunkTokens = 256;

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

int gcd(int a, int b) {
    while (b != 0) {
        const int remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

std::size_t checked_multiply(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string(what) + " overflows size_t");
    }
    return a * b;
}

#if RDMA_PROXY_HAVE_CUDA
void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}
#endif

}  // namespace

RouterRouting::RouterRouting(ProxyConfig config) : config_(std::move(config)) {}

RouterRouting::~RouterRouting() {
    release_cuda();
}

int RouterRouting::experts_per_gpu() const {
    return config_.router_num_experts / (config_.num_nodes * config_.num_gpus_per_node);
}

void RouterRouting::initialize() {
    if (initialized_ || !config_.router_routing_enabled) return;
    token_indices_by_node_.assign(static_cast<std::size_t>(config_.num_nodes), {});
    token_masks_by_node_.assign(static_cast<std::size_t>(config_.num_nodes), {});
    expert_metadata_by_gpu_.assign(
        static_cast<std::size_t>(config_.num_nodes * config_.num_gpus_per_node), {});
    if (config_.mock_mode) {
        initialize_mock();
    } else {
        initialize_cuda();
    }
    initialized_ = true;
    for (int node = 0; node < config_.num_nodes; ++node) {
        RDMA_PROXY_LOG_INFO(
            "router x3/x4 ready node=", node,
            " tokens=", token_indices_by_node_[static_cast<std::size_t>(node)].size(),
            node == config_.node_rank ? " (local node ignored for RDMA)" : "");
    }
}

const std::vector<std::size_t>& RouterRouting::token_indices_for_node(int node_rank) const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (node_rank < 0 || node_rank >= config_.num_nodes) {
        throw std::runtime_error("router node rank out of range");
    }
    return token_indices_by_node_[static_cast<std::size_t>(node_rank)];
}

const std::vector<uint8_t>& RouterRouting::token_masks_for_node(int node_rank) const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (node_rank < 0 || node_rank >= config_.num_nodes) {
        throw std::runtime_error("router node rank out of range");
    }
    return token_masks_by_node_[static_cast<std::size_t>(node_rank)];
}

const int32_t* RouterRouting::device_token_indices_for_node(int node_rank) const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (node_rank < 0 || node_rank >= config_.num_nodes) {
        throw std::runtime_error("router node rank out of range");
    }
    if (config_.mock_mode) return nullptr;
#if RDMA_PROXY_HAVE_CUDA
    if (!device_x3_ || !pinned_node_offsets_) {
        throw std::runtime_error("router device x3 is not initialized");
    }
    return device_x3_ + pinned_node_offsets_[node_rank];
#else
    throw std::runtime_error("router device x3 requested without CUDA support");
#endif
}

const RouterExpertMetadata& RouterRouting::expert_metadata_for_gpu(
    int node_rank,
    int local_gpu_index) const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (node_rank < 0 || node_rank >= config_.num_nodes ||
        local_gpu_index < 0 || local_gpu_index >= config_.num_gpus_per_node) {
        throw std::runtime_error("router expert metadata destination is out of range");
    }
    const auto global_gpu = static_cast<std::size_t>(
        node_rank * config_.num_gpus_per_node + local_gpu_index);
    return expert_metadata_by_gpu_[global_gpu];
}

void RouterRouting::initialize_mock() {
    const int experts_per_node = experts_per_gpu() * config_.num_gpus_per_node;
    std::vector<std::vector<std::pair<int, std::size_t>>> entries(
        static_cast<std::size_t>(config_.num_nodes));
    std::vector<std::vector<int>> routes(config_.num_tokens);

    for (std::size_t token = 0; token < config_.num_tokens; ++token) {
        uint64_t random0 = splitmix64(config_.router_seed ^ static_cast<uint64_t>(token));
        const uint64_t random1 = splitmix64(random0);
        int expert = static_cast<int>(random0 % static_cast<uint64_t>(config_.router_num_experts));
        int stride = 1;
        if (config_.router_num_experts > 1) {
            stride = 1 + static_cast<int>(
                random1 % static_cast<uint64_t>(config_.router_num_experts - 1));
            while (gcd(stride, config_.router_num_experts) != 1) {
                ++stride;
                if (stride == config_.router_num_experts) stride = 1;
            }
        }

        std::map<int, int> masks;
        for (int route = 0; route < config_.router_top_k; ++route) {
            routes[token].push_back(expert);
            const int node = expert / experts_per_node;
            const int local_gpu = (expert / experts_per_gpu()) % config_.num_gpus_per_node;
            int bit = config_.local_gpu_index - local_gpu;
            if (bit < 0) bit += config_.num_gpus_per_node;
            masks[node] |= 1 << bit;
            expert += stride;
            if (expert >= config_.router_num_experts) expert -= config_.router_num_experts;
        }
        for (const auto& [node, mask] : masks) {
            entries[static_cast<std::size_t>(node)].emplace_back(mask, token);
        }
    }

    std::vector<std::vector<int32_t>> expert_lists(
        static_cast<std::size_t>(config_.router_num_experts));

    for (int node = 0; node < config_.num_nodes; ++node) {
        auto& node_entries = entries[static_cast<std::size_t>(node)];
        std::stable_sort(node_entries.begin(), node_entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first;
        });
        auto& output = token_indices_by_node_[static_cast<std::size_t>(node)];
        auto& masks = token_masks_by_node_[static_cast<std::size_t>(node)];
        output.reserve(node_entries.size());
        masks.reserve(node_entries.size());
        for (const auto& entry : node_entries) {
            masks.push_back(static_cast<uint8_t>(entry.first));
            output.push_back(entry.second);
        }

        std::vector<int32_t> gpu_positions(
            static_cast<std::size_t>(config_.num_gpus_per_node), 0);
        for (std::size_t node_position = 0;
             node_position < node_entries.size();
             ++node_position) {
            const auto& [mask, token] = node_entries[node_position];
            for (const int expert : routes[token]) {
                if (expert / experts_per_node != node) continue;
                const int local_gpu =
                    (expert / experts_per_gpu()) % config_.num_gpus_per_node;
                expert_lists[static_cast<std::size_t>(expert)].push_back(
                    local_gpu == config_.local_gpu_index
                        ? static_cast<int32_t>(node_position)
                        : gpu_positions[static_cast<std::size_t>(local_gpu)]);
            }
            for (int gpu = 0; gpu < config_.num_gpus_per_node; ++gpu) {
                int bit = config_.local_gpu_index - gpu;
                if (bit < 0) bit += config_.num_gpus_per_node;
                if ((mask & (1 << bit)) != 0) {
                    ++gpu_positions[static_cast<std::size_t>(gpu)];
                }
            }
        }
    }

    for (int destination_node = 0; destination_node < config_.num_nodes; ++destination_node) {
        for (int destination_gpu = 0;
             destination_gpu < config_.num_gpus_per_node;
             ++destination_gpu) {
            const int global_gpu =
                destination_node * config_.num_gpus_per_node + destination_gpu;
            const int first_expert = global_gpu * experts_per_gpu();
            auto& metadata = expert_metadata_by_gpu_[static_cast<std::size_t>(global_gpu)];
            metadata.source_node_rank = config_.node_rank;
            metadata.source_gpu_index = config_.local_gpu_index;
            metadata.destination_node_rank = destination_node;
            metadata.destination_gpu_index = destination_gpu;
            metadata.num_nodes = config_.num_nodes;
            metadata.num_gpus_per_node = config_.num_gpus_per_node;
            metadata.num_experts = config_.router_num_experts;
            metadata.experts_per_gpu = experts_per_gpu();
            metadata.first_global_expert = first_expert;
            metadata.num_tokens = config_.num_tokens;
            metadata.expert_offsets.push_back(0);
            for (int local_expert = 0; local_expert < experts_per_gpu(); ++local_expert) {
                const auto& list = expert_lists[static_cast<std::size_t>(
                    first_expert + local_expert)];
                metadata.expert_token_indices.insert(
                    metadata.expert_token_indices.end(), list.begin(), list.end());
                metadata.expert_offsets.push_back(static_cast<int32_t>(
                    metadata.expert_token_indices.size()));
            }
        }
    }
}

void RouterRouting::initialize_cuda() {
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaSetDevice(config_.cuda_device_id), "cudaSetDevice for router routing");
    const std::size_t num_tokens = config_.num_tokens;
    const std::size_t top_k = static_cast<std::size_t>(config_.router_top_k);
    const std::size_t num_experts = static_cast<std::size_t>(config_.router_num_experts);
    const std::size_t num_nodes = static_cast<std::size_t>(config_.num_nodes);
    const std::size_t num_masks = std::size_t{1} << config_.num_gpus_per_node;
    const std::size_t num_input_chunks =
        (num_tokens + kKernelChunkTokens - 1) / kKernelChunkTokens;
    const std::size_t input_num_bins = num_experts + num_nodes * num_masks;
    const std::size_t max_node_tokens =
        checked_multiply(num_tokens, std::min(top_k, num_nodes), "router x3 capacity");

    auto allocate = [&](std::size_t bytes, const char* what) -> void* {
        void* ptr = nullptr;
        check_cuda(cudaMalloc(&ptr, bytes), what);
        device_allocations_.push_back(ptr);
        return ptr;
    };

    auto* r = static_cast<int32_t*>(allocate(
        checked_multiply(checked_multiply(num_tokens, top_k, "R elements"), sizeof(int32_t), "R bytes"),
        "cudaMalloc R"));
    auto* expert_offsets = static_cast<int32_t*>(allocate(
        (num_experts + 1) * sizeof(int32_t), "cudaMalloc expert offsets"));
    auto* expert_token_indices = static_cast<int32_t*>(allocate(
        checked_multiply(checked_multiply(num_tokens, top_k, "expert index elements"),
                         sizeof(int32_t), "expert index bytes"),
        "cudaMalloc expert token indices"));
    auto* x3 = static_cast<int32_t*>(allocate(
        max_node_tokens * sizeof(int32_t), "cudaMalloc x3"));
    device_x3_ = x3;
    auto* x4 = static_cast<uint8_t*>(allocate(max_node_tokens * sizeof(uint8_t), "cudaMalloc x4"));
    auto* node_offsets = static_cast<int32_t*>(allocate(
        (num_nodes + 1) * sizeof(int32_t), "cudaMalloc node offsets"));
    const std::size_t input_scratch_elements =
        checked_multiply(num_input_chunks, input_num_bins, "router input scratch");
    auto* input_counts = static_cast<int32_t*>(allocate(
        input_scratch_elements * sizeof(int32_t), "cudaMalloc input counts"));
    auto* input_prefixes = static_cast<int32_t*>(allocate(
        input_scratch_elements * sizeof(int32_t), "cudaMalloc input prefixes"));
    auto* mask_offsets = static_cast<int32_t*>(allocate(
        num_nodes * num_masks * sizeof(int32_t), "cudaMalloc node mask offsets"));
    const std::size_t reordered_num_bins =
        static_cast<std::size_t>(experts_per_gpu() * config_.num_gpus_per_node +
                                 config_.num_gpus_per_node);
    const std::size_t reordered_scratch_elements = checked_multiply(
        checked_multiply(num_nodes, num_input_chunks, "router reordered scratch chunks"),
        reordered_num_bins, "router reordered scratch");
    auto* reordered_counts = static_cast<int32_t*>(allocate(
        reordered_scratch_elements * sizeof(int32_t), "cudaMalloc reordered counts"));
    auto* reordered_prefixes = static_cast<int32_t*>(allocate(
        reordered_scratch_elements * sizeof(int32_t), "cudaMalloc reordered prefixes"));

    check_cuda(cudaMallocHost(
        reinterpret_cast<void**>(&pinned_node_offsets_),
        (num_nodes + 1) * sizeof(int32_t)), "cudaMallocHost node offsets");
    check_cuda(cudaMallocHost(
        reinterpret_cast<void**>(&pinned_x3_),
        max_node_tokens * sizeof(int32_t)), "cudaMallocHost x3");
    check_cuda(cudaMallocHost(
        reinterpret_cast<void**>(&pinned_x4_),
        max_node_tokens * sizeof(uint8_t)), "cudaMallocHost x4");

    generate_r_cuda_raw(
        r, static_cast<int>(num_tokens), config_.router_top_k,
        config_.router_num_experts, config_.router_seed, nullptr);
    get_expert_token_idx_node_mask_cuda_raw(
        r, static_cast<int>(num_tokens), config_.router_top_k,
        config_.router_num_experts, experts_per_gpu(), config_.num_gpus_per_node,
        config_.local_gpu_index, expert_token_indices, expert_offsets, x3, x4,
        node_offsets, input_counts, input_prefixes, mask_offsets,
        reordered_counts, reordered_prefixes, nullptr);

    check_cuda(cudaMemcpy(
        pinned_node_offsets_, node_offsets, (num_nodes + 1) * sizeof(int32_t),
        cudaMemcpyDeviceToHost), "cudaMemcpy D2H router node offsets");
    const int32_t total_x3 = pinned_node_offsets_[num_nodes];
    if (total_x3 < 0 || static_cast<std::size_t>(total_x3) > max_node_tokens) {
        throw std::runtime_error("router kernel returned an invalid x3 length");
    }
    check_cuda(cudaMemcpy(
        pinned_x3_, x3, static_cast<std::size_t>(total_x3) * sizeof(int32_t),
        cudaMemcpyDeviceToHost), "cudaMemcpy D2H router x3 to pinned memory");
    check_cuda(cudaMemcpy(
        pinned_x4_, x4, static_cast<std::size_t>(total_x3) * sizeof(uint8_t),
        cudaMemcpyDeviceToHost), "cudaMemcpy D2H router x4 to pinned memory");

    std::vector<int32_t> host_expert_offsets(num_experts + 1);
    std::vector<int32_t> host_expert_token_indices(num_tokens * top_k);
    check_cuda(cudaMemcpy(
        host_expert_offsets.data(), expert_offsets,
        host_expert_offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H router expert offsets");
    check_cuda(cudaMemcpy(
        host_expert_token_indices.data(), expert_token_indices,
        host_expert_token_indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H router expert token indices");

    for (std::size_t global_gpu = 0;
         global_gpu < num_nodes * static_cast<std::size_t>(config_.num_gpus_per_node);
         ++global_gpu) {
        const int first_expert = static_cast<int>(global_gpu) * experts_per_gpu();
        const int32_t begin = host_expert_offsets[static_cast<std::size_t>(first_expert)];
        const int32_t end = host_expert_offsets[
            static_cast<std::size_t>(first_expert + experts_per_gpu())];
        if (begin < 0 || end < begin ||
            static_cast<std::size_t>(end) > host_expert_token_indices.size()) {
            throw std::runtime_error("router kernel returned invalid expert metadata offsets");
        }
        auto& metadata = expert_metadata_by_gpu_[global_gpu];
        metadata.source_node_rank = config_.node_rank;
        metadata.source_gpu_index = config_.local_gpu_index;
        metadata.destination_node_rank =
            static_cast<int>(global_gpu) / config_.num_gpus_per_node;
        metadata.destination_gpu_index =
            static_cast<int>(global_gpu) % config_.num_gpus_per_node;
        metadata.num_nodes = config_.num_nodes;
        metadata.num_gpus_per_node = config_.num_gpus_per_node;
        metadata.num_experts = config_.router_num_experts;
        metadata.experts_per_gpu = experts_per_gpu();
        metadata.first_global_expert = first_expert;
        metadata.num_tokens = num_tokens;
        metadata.expert_token_indices.assign(
            host_expert_token_indices.begin() + begin,
            host_expert_token_indices.begin() + end);
        metadata.expert_offsets.reserve(
            static_cast<std::size_t>(experts_per_gpu()) + 1);
        for (int local_expert = 0; local_expert <= experts_per_gpu(); ++local_expert) {
            metadata.expert_offsets.push_back(
                host_expert_offsets[static_cast<std::size_t>(
                    first_expert + local_expert)] - begin);
        }
    }

    for (std::size_t node = 0; node < num_nodes; ++node) {
        const int32_t begin = pinned_node_offsets_[node];
        const int32_t end = pinned_node_offsets_[node + 1];
        if (begin < 0 || end < begin || end > total_x3) {
            throw std::runtime_error("router kernel returned invalid per-node x3 offsets");
        }
        auto& output = token_indices_by_node_[node];
        auto& masks = token_masks_by_node_[node];
        output.reserve(static_cast<std::size_t>(end - begin));
        masks.reserve(static_cast<std::size_t>(end - begin));
        for (int32_t position = begin; position < end; ++position) {
            const int32_t token = pinned_x3_[position];
            if (token < 0 || static_cast<std::size_t>(token) >= num_tokens) {
                throw std::runtime_error("router kernel returned an out-of-range token index");
            }
            output.push_back(static_cast<std::size_t>(token));
            const auto mask = pinned_x4_[position];
            const unsigned meaningful_mask = config_.num_gpus_per_node == 8 ?
                0xffU : ((1U << config_.num_gpus_per_node) - 1U);
            if (mask == 0 || (static_cast<unsigned>(mask) & ~meaningful_mask) != 0) {
                throw std::runtime_error("router kernel returned an invalid x4 GPU mask");
            }
            masks.push_back(mask);
        }
    }
#else
    throw std::runtime_error("router routing requires a CUDA build unless mock_mode=true");
#endif
}

void RouterRouting::release_cuda() noexcept {
#if RDMA_PROXY_HAVE_CUDA
    if (pinned_x3_) cudaFreeHost(pinned_x3_);
    if (pinned_x4_) cudaFreeHost(pinned_x4_);
    if (pinned_node_offsets_) cudaFreeHost(pinned_node_offsets_);
    pinned_x3_ = nullptr;
    pinned_x4_ = nullptr;
    pinned_node_offsets_ = nullptr;
    device_x3_ = nullptr;
    for (auto it = device_allocations_.rbegin(); it != device_allocations_.rend(); ++it) {
        cudaFree(*it);
    }
#endif
    device_allocations_.clear();
}

}  // namespace rdma_proxy
