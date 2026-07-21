#include "cuda_buffers.hpp"
#include "forward_computation.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed with status " + std::to_string(status));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool full = argc > 1 && std::string(argv[1]) == "--full";
        rdma_proxy::ProxyConfig config;
        config.node_rank = 0;
        config.num_nodes = 2;
        config.local_gpu_index = 0;
        config.num_gpus_per_node = 2;
        config.num_tokens = full ? 256 : 65;
        config.token_dimension = full ? 4096 : 64;
        config.tokens_per_chunk = 16;
        config.num_qps_per_peer = 1;
        config.dtype = rdma_proxy::DataType::kBF16;
        config.cuda_device_id = 0;
        config.mock_mode = false;
        config.nvlink_forwarding_enabled = true;
        config.nvlink_forward_threshold_tokens = config.num_tokens;
        config.nvlink_forward_chunk_tokens = config.num_tokens;
        config.nvlink_forward_synchronize_batches = true;
        config.nvlink_forward_completion_notifications_enabled = true;
        config.nvlink_forward_computation_enabled = true;
        config.nvlink_forward_computation_output_dim = full ? 6400 : 128;
        config.nvlink_forward_computation_tile_m = full ? 128 : 32;
        config.nvlink_forward_computation_tile_n = full ? 128 : 64;
        config.nvlink_forward_computation_num_queues = full ? 8 : 2;
        config.nvlink_forward_computation_queue_depth = full ? 128 : 8;
        config.completion_timeout_ms = full ? 120000 : 30000;
        config.peers.push_back(rdma_proxy::PeerAddress{1, "validation", 1});
        rdma_proxy::validate_config(config);

        rdma_proxy::CudaBuffers buffers(config);
        buffers.initialize();
        const std::size_t a_elements = config.num_tokens * config.token_dimension;
        std::vector<__nv_bfloat16> host_a(a_elements);
        for (std::size_t i = 0; i < a_elements; ++i) {
            const int bucket = static_cast<int>((i * 17 + 3) % 31) - 15;
            host_a[i] = __float2bfloat16_rn(static_cast<float>(bucket) / 32.0F);
        }
        const auto& receive = buffers.nvlink_receive_buffer_for_source(1);
        check_cuda(cudaMemcpy(
                       receive.recv.ptr,
                       host_a.data(),
                       host_a.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpy validation A");

        rdma_proxy::ForwardComputation computation(config, buffers);
        computation.initialize();
        computation.begin_iteration(0);
        const std::size_t first_rows = config.num_tokens / 2;
        const std::size_t row_bytes = config.token_dimension * sizeof(__nv_bfloat16);
        const auto first_tasks = computation.enqueue_ready_region(0, 1, 0, 0, first_rows, 0);
        if (!computation.wait_until_completed(first_tasks, std::chrono::milliseconds(config.completion_timeout_ms))) {
            throw std::runtime_error("first notification did not compute before final notification");
        }
        computation.enqueue_ready_region(
            0, 1, 0, first_rows, config.num_tokens - first_rows, first_rows * row_bytes);
        computation.finish_iteration(0);
        const auto stats = computation.stats();
        auto sm_ids = stats.physical_sm_ids;
        std::sort(sm_ids.begin(), sm_ids.end());
        const auto unique_sm_end = std::unique(sm_ids.begin(), sm_ids.end());
        const auto unique_sm_count = static_cast<std::size_t>(std::distance(sm_ids.begin(), unique_sm_end));

        const std::size_t d_elements =
            config.num_tokens * config.nvlink_forward_computation_output_dim;
        __nv_bfloat16* reference_device = nullptr;
        check_cuda(cudaMalloc(&reference_device, d_elements * sizeof(__nv_bfloat16)),
                   "cudaMalloc cuBLAS reference D");
        cublasHandle_t handle = nullptr;
        check_cublas(cublasCreate(&handle), "cublasCreate");
        check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const int m = static_cast<int>(config.num_tokens);
        const int n = static_cast<int>(config.nvlink_forward_computation_output_dim);
        const int k = static_cast<int>(config.token_dimension);
        // Row-major D=A*B is column-major D^T=B^T*A^T with dimensions N x M.
        check_cublas(cublasGemmEx(
                         handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         n,
                         m,
                         k,
                         &alpha,
                         buffers.nvlink_computation_weight_buffer().ptr,
                         CUDA_R_16BF,
                         n,
                         receive.recv.ptr,
                         CUDA_R_16BF,
                         k,
                         &beta,
                         reference_device,
                         CUDA_R_16BF,
                         n,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     "cublasGemmEx BF16 reference");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize cuBLAS reference");

        std::vector<__nv_bfloat16> expected(d_elements);
        std::vector<__nv_bfloat16> actual(d_elements);
        check_cuda(cudaMemcpy(
                       expected.data(), reference_device, expected.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy cuBLAS reference D");
        const auto& output = buffers.nvlink_computation_output_buffer_for_source(1);
        check_cuda(cudaMemcpy(
                       actual.data(), output.output.ptr, actual.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy persistent D");

        std::size_t mismatches = 0;
        float max_absolute_error = 0.0F;
        for (std::size_t i = 0; i < d_elements; ++i) {
            const float reference = __bfloat162float(expected[i]);
            const float observed = __bfloat162float(actual[i]);
            const float error = std::fabs(reference - observed);
            max_absolute_error = std::max(max_absolute_error, error);
            const float tolerance = std::max(0.125F, std::fabs(reference) * 0.05F);
            if (!std::isfinite(observed) || error > tolerance) ++mismatches;
        }

        std::cout << "persistent_forward_validation mode=" << (full ? "full" : "quick")
                  << " M=" << m << " K=" << k << " N=" << n
                  << " generated=" << stats.generated_tasks
                  << " completed=" << stats.tasks_completed
                  << " exits=" << stats.exit_tasks_consumed
                  << " physical_sms_observed=" << unique_sm_count
                  << " queue_full_stalls=" << stats.queue_full_stalls
                  << " poll_iterations=" << stats.poll_iterations
                  << " elapsed_ms=" << stats.iteration_seconds * 1.0e3
                  << " enqueue_ms=" << stats.enqueue_seconds * 1.0e3
                  << " first_notification_completed_before_final=true"
                  << " max_absolute_error=" << max_absolute_error
                  << " mismatches=" << mismatches << '\n';

        cublasDestroy(handle);
        cudaFree(reference_device);
        computation.shutdown();
        if (mismatches != 0) return 2;

        auto load_only_config = config;
        load_only_config.nvlink_forward_computation_load_only_enabled = true;
        rdma_proxy::validate_config(load_only_config);
        rdma_proxy::ForwardComputation load_only_computation(load_only_config, buffers);
        load_only_computation.initialize();
        load_only_computation.begin_iteration(1);
        const auto load_only_tasks = load_only_computation.enqueue_ready_region(
            1, 1, 0, 0, config.num_tokens, 0);
        load_only_computation.finish_iteration(1);
        const auto load_only_stats = load_only_computation.stats();
        check_cuda(cudaMemcpy(
                       actual.data(), output.output.ptr, actual.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy load-only D");
        std::size_t load_only_nonzero_outputs = 0;
        for (const auto value : actual) {
            if (__bfloat162float(value) != 0.0F) ++load_only_nonzero_outputs;
        }
        std::cout << "persistent_forward_load_only_validation mode=" << (full ? "full" : "quick")
                  << " generated=" << load_only_stats.generated_tasks
                  << " completed=" << load_only_stats.tasks_completed
                  << " exits=" << load_only_stats.exit_tasks_consumed
                  << " expected_tasks=" << load_only_tasks
                  << " nonzero_outputs=" << load_only_nonzero_outputs << '\n';
        load_only_computation.shutdown();
        if (load_only_stats.tasks_completed != load_only_tasks || load_only_nonzero_outputs != 0) return 3;

        auto dequeue_only_config = config;
        dequeue_only_config.nvlink_forward_computation_dequeue_only_enabled = true;
        rdma_proxy::validate_config(dequeue_only_config);
        rdma_proxy::ForwardComputation dequeue_only_computation(dequeue_only_config, buffers);
        dequeue_only_computation.initialize();
        dequeue_only_computation.begin_iteration(2);
        const auto dequeue_only_tasks = dequeue_only_computation.enqueue_ready_region(
            2, 1, 0, 0, config.num_tokens, 0);
        dequeue_only_computation.finish_iteration(2);
        const auto dequeue_only_stats = dequeue_only_computation.stats();
        check_cuda(cudaMemcpy(
                       actual.data(), output.output.ptr, actual.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy dequeue-only D");
        std::size_t dequeue_only_nonzero_outputs = 0;
        for (const auto value : actual) {
            if (__bfloat162float(value) != 0.0F) ++dequeue_only_nonzero_outputs;
        }
        std::cout << "persistent_forward_dequeue_only_validation mode=" << (full ? "full" : "quick")
                  << " generated=" << dequeue_only_stats.generated_tasks
                  << " completed=" << dequeue_only_stats.tasks_completed
                  << " exits=" << dequeue_only_stats.exit_tasks_consumed
                  << " expected_tasks=" << dequeue_only_tasks
                  << " nonzero_outputs=" << dequeue_only_nonzero_outputs << '\n';
        dequeue_only_computation.shutdown();
        if (dequeue_only_stats.tasks_completed != dequeue_only_tasks ||
            dequeue_only_nonzero_outputs != 0) return 4;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "validate_forward_computation failed: " << error.what() << '\n';
        return 1;
    }
}
