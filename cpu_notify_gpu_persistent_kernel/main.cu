#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using Clock = std::chrono::steady_clock;
static_assert(sizeof(std::uint32_t) == 4, "the control protocol requires 4-byte values");

std::runtime_error cuda_runtime_error(cudaError_t status, const char* expression,
                                      const char* file, int line) {
  std::ostringstream message;
  message << expression << " failed at " << file << ':' << line << ": "
          << cudaGetErrorName(status) << " (" << cudaGetErrorString(status) << ')';
  return std::runtime_error(message.str());
}

std::runtime_error cuda_driver_error(CUresult status, const char* expression,
                                     const char* file, int line) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &description);

  std::ostringstream message;
  message << expression << " failed at " << file << ':' << line << ": "
          << (name != nullptr ? name : "unknown CUDA driver error") << " ("
          << (description != nullptr ? description : "no description") << ')';
  return std::runtime_error(message.str());
}

#define CUDA_CHECK(expression)                                                \
  do {                                                                        \
    const cudaError_t status_ = (expression);                                 \
    if (status_ != cudaSuccess) {                                             \
      throw cuda_runtime_error(status_, #expression, __FILE__, __LINE__);     \
    }                                                                         \
  } while (false)

#define CU_CHECK(expression)                                                  \
  do {                                                                        \
    const CUresult status_ = (expression);                                    \
    if (status_ != CUDA_SUCCESS) {                                            \
      throw cuda_driver_error(status_, #expression, __FILE__, __LINE__);      \
    }                                                                         \
  } while (false)

struct Config {
  std::uint64_t period_us = 1000;
  std::uint32_t increment = 3;
  std::uint32_t threshold = 1000;
  int device = 0;
  int threads_per_cta = 32;
};

[[noreturn]] void usage_error(const std::string& message) {
  throw std::invalid_argument(message + "\nUse --help to see valid arguments.");
}

std::uint64_t parse_u64(const std::string& text, const char* option) {
  if (text.empty() || text.front() == '-') {
    usage_error(std::string(option) + " expects a non-negative integer");
  }

  std::size_t consumed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &consumed, 10);
  } catch (const std::exception&) {
    usage_error(std::string(option) + " expects an integer, got '" + text + "'");
  }
  if (consumed != text.size()) {
    usage_error(std::string(option) + " expects an integer, got '" + text + "'");
  }
  return static_cast<std::uint64_t>(value);
}

std::string option_value(int& index, int argc, char** argv,
                         const std::string& argument) {
  const std::size_t equals = argument.find('=');
  if (equals != std::string::npos) {
    return argument.substr(equals + 1);
  }
  if (++index >= argc) {
    usage_error(argument + " requires a value");
  }
  return argv[index];
}

bool is_option(const std::string& argument, const std::string& long_name,
               const std::string& short_name) {
  return argument == long_name || argument == short_name ||
         argument.rfind(long_name + '=', 0) == 0 ||
         argument.rfind(short_name + '=', 0) == 0;
}

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Launch one persistent CTA per SM. The CPU increments a pinned 32-bit\n"
      << "control value every t microseconds and publishes it to device memory\n"
      << "with cuStreamWriteValue32.\n\n"
      << "Options:\n"
      << "  --period-us, --t N       Update period t in microseconds (default: 1000)\n"
      << "  --increment, --b N       Increment b (default: 3)\n"
      << "  --threshold, --w N       Termination threshold w (default: 1000)\n"
      << "  --device N               CUDA device ordinal (default: 0)\n"
      << "  --threads-per-cta N      Threads in each CTA (default: 32)\n"
      << "  --help, -h               Show this help\n\n"
      << "The final CPU value must be strictly greater than w. If repeated additions\n"
      << "of b land exactly on w, a CTA with j == w cannot satisfy control > j, so\n"
      << "that configuration is rejected instead of hanging.\n";
}

Config parse_arguments(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      print_help(argv[0]);
      std::exit(EXIT_SUCCESS);
    }

    if (is_option(argument, "--period-us", "--t")) {
      config.period_us = parse_u64(option_value(i, argc, argv, argument), "--period-us");
    } else if (is_option(argument, "--increment", "--b")) {
      const auto value = parse_u64(option_value(i, argc, argv, argument), "--increment");
      if (value > std::numeric_limits<std::uint32_t>::max()) {
        usage_error("--increment does not fit in a 4-byte unsigned integer");
      }
      config.increment = static_cast<std::uint32_t>(value);
    } else if (is_option(argument, "--threshold", "--w")) {
      const auto value = parse_u64(option_value(i, argc, argv, argument), "--threshold");
      if (value > std::numeric_limits<std::uint32_t>::max()) {
        usage_error("--threshold does not fit in a 4-byte unsigned integer");
      }
      config.threshold = static_cast<std::uint32_t>(value);
    } else if (is_option(argument, "--device", "--device")) {
      const auto value = parse_u64(option_value(i, argc, argv, argument), "--device");
      if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        usage_error("--device is too large");
      }
      config.device = static_cast<int>(value);
    } else if (is_option(argument, "--threads-per-cta", "--threads-per-cta")) {
      const auto value = parse_u64(option_value(i, argc, argv, argument),
                                   "--threads-per-cta");
      if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        usage_error("--threads-per-cta is too large");
      }
      config.threads_per_cta = static_cast<int>(value);
    } else {
      usage_error("unknown option '" + argument + "'");
    }
  }
  return config;
}

void validate_config(const Config& config) {
  if (config.period_us == 0) {
    usage_error("--period-us must be greater than zero");
  }
  if (config.period_us >
      static_cast<std::uint64_t>(std::chrono::microseconds::max().count())) {
    usage_error("--period-us is too large");
  }
  if (config.increment == 0) {
    usage_error("--increment must be greater than zero");
  }
  if (config.threshold == 0) {
    usage_error("--threshold must be greater than zero");
  }
  if (config.threads_per_cta <= 0 || config.threads_per_cta > 1024) {
    usage_error("--threads-per-cta must be in [1, 1024]");
  }

  const std::uint64_t updates =
      (static_cast<std::uint64_t>(config.threshold) + config.increment - 1) /
      config.increment;
  const std::uint64_t final_value = updates * config.increment;
  if (final_value > std::numeric_limits<std::uint32_t>::max()) {
    usage_error("the final control value would overflow its 4-byte storage");
  }
  if (final_value == config.threshold) {
    usage_error("this b/w pair stops at control == w and would deadlock; choose an "
                "increment that makes the final value overshoot w");
  }
}

// Only thread 0 is needed for the requested CTA-level state machine. The other
// threads, if configured, remain part of the resident CTA but return immediately.
// dynamic_shared[0..3] is the CTA's shared-memory copy of j. The deliberately
// large dynamic allocation selected by the host limits occupancy to one CTA/SM.
__global__ void persistent_poll_kernel(const volatile std::uint32_t* control,
                                       std::uint32_t* results,
                                       std::uint32_t threshold,
                                       std::uint32_t sm_count) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  volatile auto* j = reinterpret_cast<std::uint32_t*>(dynamic_shared);

  if (threadIdx.x != 0) {
    return;
  }

  *j = blockIdx.x;
  while (true) {
    const std::uint32_t observed = *control;
    if (observed > *j) {
      *j += sm_count;
    }

    if (*j > threshold) {
      results[blockIdx.x] = *j;
      return;
    }
  }
}

std::size_t single_cta_shared_memory(int device, int threads_per_cta) {
  int max_opt_in = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_opt_in,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    device));
  if (max_opt_in < static_cast<int>(sizeof(std::uint32_t))) {
    throw std::runtime_error("device cannot provide enough dynamic shared memory");
  }

  CUDA_CHECK(cudaFuncSetAttribute(
      persistent_poll_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_opt_in));
  CUDA_CHECK(cudaFuncSetAttribute(
      persistent_poll_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

  constexpr std::size_t allocation_granularity = 256;
  for (std::size_t bytes = allocation_granularity;
       bytes <= static_cast<std::size_t>(max_opt_in);
       bytes += allocation_granularity) {
    int active_ctas = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_ctas, persistent_poll_kernel, threads_per_cta, bytes));
    if (active_ctas == 1) {
      CUDA_CHECK(cudaFuncSetAttribute(
          persistent_poll_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(bytes)));
      return bytes;
    }
  }

  throw std::runtime_error(
      "could not find a dynamic shared-memory size that limits occupancy to one CTA/SM");
}

std::uint32_t expected_result(std::uint32_t cta, std::uint32_t sm_count,
                              std::uint32_t threshold) {
  if (cta > threshold) {
    return cta;
  }
  const std::uint64_t steps =
      (static_cast<std::uint64_t>(threshold) - cta) / sm_count + 1;
  return static_cast<std::uint32_t>(cta + steps * sm_count);
}

int run(const Config& config) {
  CUDA_CHECK(cudaSetDevice(config.device));
  CUDA_CHECK(cudaFree(nullptr));  // Establish the runtime/driver primary context.

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, config.device));
  if (properties.major < 9) {
    throw std::runtime_error("this program requires a Hopper-class (SM 9.x) or newer GPU");
  }
  if (config.threads_per_cta > properties.maxThreadsPerBlock) {
    throw std::runtime_error("--threads-per-cta exceeds this device's block limit");
  }

  const std::uint32_t sm_count =
      static_cast<std::uint32_t>(properties.multiProcessorCount);
  if (config.threshold > std::numeric_limits<std::uint32_t>::max() - sm_count) {
    throw std::runtime_error("w is too large: a CTA's final j could overflow 4 bytes");
  }

  const std::size_t dynamic_shared_bytes =
      single_cta_shared_memory(config.device, config.threads_per_cta);

  std::uint32_t* host_control = nullptr;
  std::uint32_t* host_results = nullptr;
  std::uint32_t* device_control = nullptr;
  std::uint32_t* device_results = nullptr;
  cudaStream_t kernel_stream = nullptr;
  cudaStream_t notify_stream = nullptr;

  CUDA_CHECK(cudaMallocHost(&host_control, sizeof(*host_control)));
  CUDA_CHECK(cudaMallocHost(&host_results, sm_count * sizeof(*host_results)));
  CUDA_CHECK(cudaMalloc(&device_control, sizeof(*device_control)));
  CUDA_CHECK(cudaMalloc(&device_results, sm_count * sizeof(*device_results)));
  CUDA_CHECK(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking));

  int least_priority = 0;
  int greatest_priority = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  CUDA_CHECK(cudaStreamCreateWithPriority(
      &notify_stream, cudaStreamNonBlocking, greatest_priority));

  *host_control = 0;
  CUDA_CHECK(cudaMemcpy(device_control, host_control, sizeof(*host_control),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_results, 0xff, sm_count * sizeof(*device_results)));

  std::cout << "Device: " << properties.name << " (SM " << properties.major << '.'
            << properties.minor << ")\n"
            << "S=" << sm_count << ", CTAs=" << sm_count
            << ", threads/CTA=" << config.threads_per_cta
            << ", dynamic shared/CTA=" << dynamic_shared_bytes << " bytes\n"
            << "t=" << config.period_us << " us, b=" << config.increment
            << ", w=" << config.threshold << '\n';

  persistent_poll_kernel<<<sm_count, config.threads_per_cta,
                           dynamic_shared_bytes, kernel_stream>>>(
      device_control, device_results, config.threshold, sm_count);
  CUDA_CHECK(cudaGetLastError());

  const CUstream driver_notify_stream = reinterpret_cast<CUstream>(notify_stream);
  const CUdeviceptr driver_control = static_cast<CUdeviceptr>(
      reinterpret_cast<std::uintptr_t>(device_control));

  const auto period = std::chrono::microseconds(config.period_us);
  auto next_update = Clock::now() + period;
  std::uint64_t update_count = 0;
  while (*host_control < config.threshold) {
    std::this_thread::sleep_until(next_update);
    next_update += period;

    *host_control += config.increment;
    ++update_count;
    CU_CHECK(cuStreamWriteValue32(driver_notify_stream, driver_control,
                                  *host_control, CU_STREAM_WRITE_VALUE_DEFAULT));
  }

  // Complete the last publication before waiting for the kernel that consumes it.
  CU_CHECK(cuStreamSynchronize(driver_notify_stream));
  CUDA_CHECK(cudaStreamSynchronize(kernel_stream));

  CUDA_CHECK(cudaMemcpyAsync(host_results, device_results,
                             sm_count * sizeof(*host_results),
                             cudaMemcpyDeviceToHost, kernel_stream));
  CUDA_CHECK(cudaStreamSynchronize(kernel_stream));

  std::cout << "CPU stopped after " << update_count
            << " updates; final pinned control value=" << *host_control << "\nA = [";
  bool valid = true;
  for (std::uint32_t i = 0; i < sm_count; ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << host_results[i];
    valid = valid &&
            host_results[i] == expected_result(i, sm_count, config.threshold);
  }
  std::cout << "]\nValidation: " << (valid ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaStreamDestroy(notify_stream));
  CUDA_CHECK(cudaStreamDestroy(kernel_stream));
  CUDA_CHECK(cudaFree(device_results));
  CUDA_CHECK(cudaFree(device_control));
  CUDA_CHECK(cudaFreeHost(host_results));
  CUDA_CHECK(cudaFreeHost(host_control));
  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parse_arguments(argc, argv);
    validate_config(config);
    return run(config);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
