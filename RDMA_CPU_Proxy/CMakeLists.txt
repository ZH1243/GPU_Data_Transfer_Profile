cmake_minimum_required(VERSION 3.18)

project(RDMA_CPU_Proxy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RDMA_PROXY_REQUIRE_CUDA "Fail configuration if CUDA is not available" OFF)
option(RDMA_PROXY_REQUIRE_VERBS "Fail configuration if libibverbs is not available" OFF)
option(RDMA_PROXY_BUILD_TESTS "Build unit tests" ON)

include(CheckLanguage)
check_language(CUDA)
set(RDMA_PROXY_HAVE_CUDA 0)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(RDMA_PROXY_HAVE_CUDA 1)
elseif(RDMA_PROXY_REQUIRE_CUDA)
  message(FATAL_ERROR "CUDA compiler not found. Install CUDA or configure with -DRDMA_PROXY_REQUIRE_CUDA=OFF.")
endif()

find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h)
find_library(IBVERBS_LIBRARY ibverbs)
set(RDMA_PROXY_HAVE_VERBS 0)
if(IBVERBS_INCLUDE_DIR AND IBVERBS_LIBRARY)
  set(RDMA_PROXY_HAVE_VERBS 1)
elseif(RDMA_PROXY_REQUIRE_VERBS)
  message(FATAL_ERROR "libibverbs not found. Install rdma-core/libibverbs or configure with -DRDMA_PROXY_REQUIRE_VERBS=OFF.")
endif()

set(RDMA_PROXY_SOURCES
  src/config.cpp
  src/cpu_affinity.cpp
  src/cuda_buffers.cpp
  src/logging.cpp
  src/protocol.cpp
  src/proxy.cpp
  src/qp_worker.cpp
  src/rdma_connection.cpp
  src/rdma_context.cpp
  src/router_routing.cpp
)

if(RDMA_PROXY_HAVE_CUDA)
  set(EXPERT_TOKEN_IDX_KERNEL_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/../get_expert_token_idx_list_gpu/expert_token_idx_kernel.cu)
  set_source_files_properties(${EXPERT_TOKEN_IDX_KERNEL_SOURCE}
    PROPERTIES COMPILE_DEFINITIONS EXPERT_TOKEN_IDX_STANDALONE=1)
  list(APPEND RDMA_PROXY_SOURCES src/cuda_kernels.cu ${EXPERT_TOKEN_IDX_KERNEL_SOURCE})
endif()

add_library(rdma_cpu_proxy_lib ${RDMA_PROXY_SOURCES})
target_include_directories(rdma_cpu_proxy_lib PUBLIC include)
if(RDMA_PROXY_HAVE_CUDA)
  target_include_directories(rdma_cpu_proxy_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../get_expert_token_idx_list_gpu)
endif()
target_compile_definitions(rdma_cpu_proxy_lib
  PUBLIC
    RDMA_PROXY_HAVE_CUDA=$<BOOL:${RDMA_PROXY_HAVE_CUDA}>
    RDMA_PROXY_HAVE_VERBS=$<BOOL:${RDMA_PROXY_HAVE_VERBS}>
)

if(RDMA_PROXY_HAVE_VERBS)
  target_include_directories(rdma_cpu_proxy_lib PRIVATE ${IBVERBS_INCLUDE_DIR})
  target_link_libraries(rdma_cpu_proxy_lib PRIVATE ${IBVERBS_LIBRARY})
endif()

if(RDMA_PROXY_HAVE_CUDA)
  set_target_properties(rdma_cpu_proxy_lib PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_link_libraries(rdma_cpu_proxy_lib PRIVATE CUDA::cudart)
endif()

find_package(Threads REQUIRED)
target_link_libraries(rdma_cpu_proxy_lib PUBLIC Threads::Threads)

add_executable(rdma_cpu_proxy src/main.cpp)
target_link_libraries(rdma_cpu_proxy PRIVATE rdma_cpu_proxy_lib)

if(RDMA_PROXY_BUILD_TESTS)
  enable_testing()

  add_executable(test_chunking tests/test_chunking.cpp)
  target_link_libraries(test_chunking PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_chunking COMMAND test_chunking)

  add_executable(test_config tests/test_config.cpp)
  target_link_libraries(test_config PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_config COMMAND test_config ${CMAKE_CURRENT_SOURCE_DIR}/config/example_config.json)

  add_executable(test_measured_run tests/test_measured_run.cpp)
  target_link_libraries(test_measured_run PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_measured_run COMMAND test_measured_run)

  add_executable(test_router_routing tests/test_router_routing.cpp)
  target_link_libraries(test_router_routing PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_router_routing COMMAND test_router_routing)

  add_executable(test_global_metadata_exchange tests/test_global_metadata_exchange.cpp)
  target_link_libraries(test_global_metadata_exchange PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_global_metadata_exchange COMMAND test_global_metadata_exchange)
endif()

message(STATUS "RDMA_PROXY_HAVE_CUDA=${RDMA_PROXY_HAVE_CUDA}")
message(STATUS "RDMA_PROXY_HAVE_VERBS=${RDMA_PROXY_HAVE_VERBS}")
