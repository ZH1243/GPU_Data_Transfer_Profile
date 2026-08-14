cmake_minimum_required(VERSION 3.24)

# Set Hopper as the default before CUDA compiler detection. A command-line
# -DCMAKE_CUDA_ARCHITECTURES=... value takes precedence.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

project(cpu_notify_gpu_persistent_kernel LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_executable(cpu_notify_gpu_persistent_kernel main.cu)
target_compile_features(cpu_notify_gpu_persistent_kernel PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(cpu_notify_gpu_persistent_kernel PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
  Threads::Threads
)

set_target_properties(cpu_notify_gpu_persistent_kernel PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(cpu_notify_gpu_persistent_kernel PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
)
