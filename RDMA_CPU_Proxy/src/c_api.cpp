#include "rdma_proxy_c_api.h"

#include "config.hpp"
#include "cpu_affinity.hpp"
#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

struct RdmaProxyHandle {
    explicit RdmaProxyHandle(rdma_proxy::ProxyConfig value)
        : config(std::move(value)), proxy(config) {}

    rdma_proxy::ProxyConfig config;
    rdma_proxy::Proxy proxy;
    bool initialized{false};
    bool shut_down{false};
    bool finished{false};
    uint64_t next_iteration{0};
};

namespace {

thread_local std::string last_error;

void clear_error() {
    last_error.clear();
}

int record_current_exception() noexcept {
    try {
        throw;
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception";
    }
    return -1;
}

std::vector<char*> make_mutable_argv(
    int argc,
    const char* const* argv,
    std::vector<std::string>& storage) {
    if (argc < 0) {
        throw std::runtime_error("argc must be non-negative");
    }
    if (argc != 0 && argv == nullptr) {
        throw std::runtime_error("argv must be non-null when argc is nonzero");
    }

    storage.clear();
    storage.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        if (argv[i] == nullptr) {
            throw std::runtime_error("argv contains a null entry");
        }
        storage.emplace_back(argv[i]);
    }

    std::vector<char*> mutable_argv;
    mutable_argv.reserve(storage.size());
    for (auto& argument : storage) {
        mutable_argv.push_back(argument.data());
    }
    return mutable_argv;
}

void select_proxy_cuda_device(const RdmaProxyHandle& handle) {
#if RDMA_PROXY_HAVE_CUDA
    if (handle.config.mock_mode) return;
    const auto status = cudaSetDevice(handle.config.cuda_device_id);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            "cudaSetDevice failed in proxy iteration coordinator: " +
            std::string(cudaGetErrorString(status)));
    }
#else
    (void)handle;
#endif
}

RdmaProxyDeviceBufferRequest make_c_device_buffer_request(
    const rdma_proxy::DeviceBufferAllocationRequest& request) {
    if (request.dimensions.size() > 2) {
        throw std::runtime_error(
            "external device-buffer request has more than two dimensions");
    }
    if (request.bytes > std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error("external device-buffer byte size exceeds C ABI range");
    }
    RdmaProxyDeviceBufferRequest result{};
    result.struct_size = sizeof(result);
    result.version = 1;
    result.kind = static_cast<int32_t>(request.kind);
    result.element_type = static_cast<int32_t>(request.element_type);
    result.peer_rank = request.peer_rank;
    result.source_node_rank = request.source_node_rank;
    result.source_gpu_index = request.source_gpu_index;
    result.dimension_count = static_cast<uint32_t>(request.dimensions.size());
    result.bytes = static_cast<uint64_t>(request.bytes);
    for (std::size_t index = 0; index < request.dimensions.size(); ++index) {
        if (request.dimensions[index] > std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error(
                "external device-buffer dimension exceeds C ABI range");
        }
        result.dimensions[index] =
            static_cast<uint64_t>(request.dimensions[index]);
    }
    return result;
}

}  // namespace

extern "C" int rdma_proxy_abi_version(void) {
    return 1;
}

extern "C" RdmaProxyHandle* rdma_proxy_create(
    int argc,
    const char* const* argv) {
    clear_error();
    try {
        std::vector<std::string> argument_storage;
        auto mutable_argv = make_mutable_argv(argc, argv, argument_storage);
        auto config = rdma_proxy::load_config(argc, mutable_argv.data());
        rdma_proxy::Logger::instance().set_level(
            rdma_proxy::log_level_from_string(config.log_level));
        rdma_proxy::apply_cpu_affinity(config);
        return new RdmaProxyHandle(std::move(config));
    } catch (...) {
        record_current_exception();
        return nullptr;
    }
}

extern "C" int rdma_proxy_set_device_buffer_allocator(
    RdmaProxyHandle* handle,
    RdmaProxyDeviceBufferAllocator allocator,
    void* context) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_set_device_buffer_allocator received a null handle";
        return -1;
    }
    if (handle->initialized) {
        last_error = "device-buffer allocator must be set before proxy initialization";
        return -1;
    }
    try {
        if (allocator == nullptr) {
            handle->proxy.set_external_device_buffer_allocator({});
        } else {
            handle->proxy.set_external_device_buffer_allocator(
                [allocator, context](
                    const rdma_proxy::DeviceBufferAllocationRequest& request) -> void* {
                    const auto c_request = make_c_device_buffer_request(request);
                    return reinterpret_cast<void*>(allocator(context, &c_request));
                });
        }
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_initialize(RdmaProxyHandle* handle) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_initialize received a null handle";
        return -1;
    }
    try {
        handle->proxy.initialize();
        handle->initialized = true;
        handle->shut_down = false;
        handle->finished = false;
        handle->next_iteration = 0;
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_configure_down_ready_trace(
    RdmaProxyHandle* handle, int32_t final_groups, uintptr_t* device_address) {
    clear_error();
    try {
        if (!handle || !device_address || !handle->initialized || handle->shut_down ||
            handle->next_iteration != 0 || handle->finished)
            throw std::runtime_error("invalid down-ready trace configuration request");
        *device_address = handle->proxy.configure_down_ready_trace(final_groups);
        return 0;
    } catch (...) { return record_current_exception(); }
}

extern "C" int rdma_proxy_read_down_ready_trace(
    RdmaProxyHandle* handle, uint64_t iteration, int32_t* groups,
    double* times_us, uint64_t capacity, uint64_t* count) {
    clear_error();
    try {
        if (!handle || !handle->initialized || handle->shut_down || !groups || !times_us || !count)
            throw std::runtime_error("invalid down-ready trace read request");
        const auto observations = handle->proxy.read_down_ready_trace(iteration);
        if (capacity < observations.size()) throw std::runtime_error("down-ready trace capacity too small");
        for (std::size_t i = 0; i < observations.size(); ++i) {
            groups[i] = observations[i].first;
            times_us[i] = observations[i].second;
        }
        *count = observations.size();
        return 0;
    } catch (...) { return record_current_exception(); }
}

extern "C" int rdma_proxy_configure_return_copies(
    RdmaProxyHandle* handle, const RdmaProxyReturnCopy* copies, uint64_t count,
    const int64_t* rows, uint64_t row_count, uint64_t width, uint64_t tile,
    uint64_t group_tiles, uint64_t element_bytes, int32_t per_group, uintptr_t down_done) {
    clear_error();
    try {
        if (!handle || !handle->initialized || handle->shut_down ||
            handle->next_iteration != 0 || handle->finished || (per_group != 0 && per_group != 1))
            throw std::runtime_error("invalid return-copy configuration request");
        handle->proxy.configure_return_copies(copies, count, rows, row_count, width,
                                             tile, group_tiles, element_bytes, per_group != 0, down_done);
        return 0;
    } catch (...) { return record_current_exception(); }
}

extern "C" int rdma_proxy_arm_return_copies(RdmaProxyHandle* handle, uint64_t iteration) {
    clear_error();
    try {
        if (!handle || !handle->initialized || handle->shut_down)
            throw std::runtime_error("invalid return-copy arm request");
        handle->proxy.arm_return_copies(iteration);
        return 0;
    } catch (...) { return record_current_exception(); }
}

extern "C" int rdma_proxy_get_return_ready_prefix(
    RdmaProxyHandle* handle, uint64_t iteration, int32_t* prefix) {
    clear_error();
    try {
        if (!handle || !handle->initialized || handle->shut_down || !prefix)
            throw std::runtime_error("invalid return-ready prefix request");
        *prefix = handle->proxy.return_ready_prefix(iteration);
        return 0;
    } catch (...) { return record_current_exception(); }
}

extern "C" int rdma_proxy_get_return_ready_publication(
    RdmaProxyHandle* handle, uintptr_t* host_address, uintptr_t* device_address) {
    clear_error();
    try {
        if (!handle || !handle->initialized || handle->shut_down || !host_address || !device_address)
            throw std::runtime_error("invalid return-ready publication request");
        const auto addresses = handle->proxy.return_ready_publication_addresses();
        *host_address = addresses.first;
        *device_address = addresses.second;
        return 0;
    } catch (...) { return record_current_exception(); }
}

extern "C" int rdma_proxy_run(RdmaProxyHandle* handle) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_run received a null handle";
        return -1;
    }
    if (!handle->initialized || handle->shut_down) {
        last_error = "rdma_proxy_run requires an initialized proxy";
        return -1;
    }
    if (handle->next_iteration != 0 || handle->finished) {
        last_error = "rdma_proxy_run cannot follow per-iteration execution";
        return -1;
    }
    try {
        select_proxy_cuda_device(*handle);
        handle->proxy.run();
        if (handle->config.num_iterations != 0) {
            handle->next_iteration = handle->config.num_iterations;
            handle->finished = true;
        }
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_get_num_iterations(
    RdmaProxyHandle* handle,
    uint64_t* num_iterations) {
    clear_error();
    if (handle == nullptr || num_iterations == nullptr) {
        last_error = "rdma_proxy_get_num_iterations received a null argument";
        return -1;
    }
    *num_iterations = static_cast<uint64_t>(handle->config.num_iterations);
    return 0;
}

extern "C" int rdma_proxy_run_iteration(
    RdmaProxyHandle* handle,
    uint64_t iteration) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_run_iteration received a null handle";
        return -1;
    }
    if (!handle->initialized || handle->shut_down) {
        last_error = "rdma_proxy_run_iteration requires an initialized proxy";
        return -1;
    }
    if (handle->config.num_iterations == 0) {
        last_error = "per-iteration execution requires finite num_iterations";
        return -1;
    }
    if (handle->finished) {
        last_error = "proxy run is already finished";
        return -1;
    }
    if (iteration != handle->next_iteration) {
        last_error = "proxy iterations must be submitted in increasing order";
        return -1;
    }
    try {
        // cudaSetDevice state is host-thread-local. Per-iteration embedding may
        // call this function from a coordinator thread other than the thread
        // that initialized the proxy, so select the configured device here.
        select_proxy_cuda_device(*handle);
        handle->proxy.run_iteration_step(iteration);
        ++handle->next_iteration;
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_prepare_iteration(
    RdmaProxyHandle* handle,
    uint64_t iteration) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_prepare_iteration received a null handle";
        return -1;
    }
    if (!handle->initialized || handle->shut_down) {
        last_error = "rdma_proxy_prepare_iteration requires an initialized proxy";
        return -1;
    }
    if (handle->config.num_iterations == 0) {
        last_error = "per-iteration execution requires finite num_iterations";
        return -1;
    }
    if (handle->finished || iteration != handle->next_iteration) {
        last_error = "only the next unfinished proxy iteration can be prepared";
        return -1;
    }
    try {
        select_proxy_cuda_device(*handle);
        handle->proxy.prepare_iteration_step(iteration);
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_record_computation_end(
    RdmaProxyHandle* handle,
    uint64_t iteration,
    uintptr_t cuda_stream) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_record_computation_end received a null handle";
        return -1;
    }
    if (!handle->initialized || handle->shut_down) {
        last_error = "recording computation end requires an initialized proxy";
        return -1;
    }
    try {
        select_proxy_cuda_device(*handle);
        handle->proxy.record_router_computation_end(
            iteration, cuda_stream);
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_get_computation_elapsed_ms(
    RdmaProxyHandle* handle,
    uint64_t iteration,
    float* elapsed_ms) {
    clear_error();
    if (handle == nullptr || elapsed_ms == nullptr) {
        last_error = "rdma_proxy_get_computation_elapsed_ms received a null argument";
        return -1;
    }
    if (!handle->initialized || handle->shut_down) {
        last_error = "querying computation time requires an initialized proxy";
        return -1;
    }
    try {
        select_proxy_cuda_device(*handle);
        *elapsed_ms = handle->proxy.router_computation_elapsed_ms(iteration);
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_get_computation_num_tokens(
    RdmaProxyHandle* handle,
    uint64_t* num_tokens) {
    clear_error();
    if (handle == nullptr || num_tokens == nullptr) {
        last_error = "rdma_proxy_get_computation_num_tokens received a null argument";
        return -1;
    }
    if (!handle->initialized || handle->shut_down) {
        last_error = "querying computation tokens requires an initialized proxy";
        return -1;
    }
    try {
        const auto count = handle->proxy.router_computation_num_tokens();
        if (count > std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error(
                "router computation token count exceeds C ABI range");
        }
        *num_tokens = static_cast<uint64_t>(count);
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_get_computation_expert_counts(
    RdmaProxyHandle* handle, uint64_t* counts, uint64_t capacity,
    uint64_t* num_experts) {
    clear_error();
    if (!handle || !num_experts || (!counts && capacity != 0)) {
        last_error = "invalid expert-count query arguments";
        return -1;
    }
    if (!handle->initialized || handle->shut_down) {
        last_error = "querying expert counts requires an initialized proxy";
        return -1;
    }
    try {
        const auto state = handle->proxy.router_computation_scheduler_state();
        if (!state.initialized) throw std::runtime_error("router scheduler is not initialized");
        *num_experts = state.expert_total_tokens.size();
        if (!counts) return 0;
        if (capacity < *num_experts) throw std::runtime_error("expert-count capacity is too small");
        for (std::size_t i = 0; i < state.expert_total_tokens.size(); ++i) {
            counts[i] = state.expert_total_tokens[i];
        }
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_get_intra_node_source_masks(
    RdmaProxyHandle* handle, int32_t source_node, uint8_t* masks,
    uint64_t capacity, uint64_t* count) {
    clear_error();
    try {
        if (!handle || !handle->initialized || handle->shut_down || !count ||
            (!masks && capacity)) throw std::runtime_error("invalid source-mask query");
        const auto values = handle->proxy.intra_node_source_masks(source_node);
        *count = values.size();
        if (!masks) return 0;
        if (capacity < values.size()) throw std::runtime_error("source-mask capacity too small");
        std::copy(values.begin(), values.end(), masks);
        return 0;
    } catch (...) { return record_current_exception(); }
}

extern "C" int rdma_proxy_finish(RdmaProxyHandle* handle) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_finish received a null handle";
        return -1;
    }
    if (!handle->initialized || handle->shut_down) {
        last_error = "rdma_proxy_finish requires an initialized proxy";
        return -1;
    }
    if (handle->config.num_iterations == 0) {
        last_error = "per-iteration execution requires finite num_iterations";
        return -1;
    }
    if (handle->next_iteration != handle->config.num_iterations) {
        last_error = "cannot finish before all configured iterations complete";
        return -1;
    }
    if (handle->finished) return 0;
    try {
        handle->proxy.finish_run();
        handle->finished = true;
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_shutdown(RdmaProxyHandle* handle) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_shutdown received a null handle";
        return -1;
    }
    try {
        handle->proxy.shutdown();
        handle->shut_down = true;
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" void rdma_proxy_destroy(RdmaProxyHandle* handle) {
    delete handle;
}

extern "C" int rdma_proxy_run_argv(
    int argc,
    const char* const* argv) {
    clear_error();
    RdmaProxyHandle* handle = rdma_proxy_create(argc, argv);
    if (handle == nullptr) return -1;

    int status = rdma_proxy_initialize(handle);
    if (status == 0) status = rdma_proxy_run(handle);

    const std::string operation_error = last_error;
    const int shutdown_status = rdma_proxy_shutdown(handle);
    const std::string shutdown_error = last_error;
    rdma_proxy_destroy(handle);

    if (status != 0) {
        last_error = operation_error;
        return status;
    }
    if (shutdown_status != 0) {
        last_error = shutdown_error;
        return shutdown_status;
    }
    clear_error();
    return 0;
}

extern "C" const char* rdma_proxy_last_error(void) {
    return last_error.c_str();
}
