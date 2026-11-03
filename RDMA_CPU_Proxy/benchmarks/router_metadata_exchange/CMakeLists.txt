cmake_minimum_required(VERSION 3.18)
project(router_metadata_benchmark LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(Threads REQUIRED)
add_library(router_metadata_bench SHARED
  native_proxy.cpp
  ../../src/router_routing.cpp
  ../../src/protocol.cpp
  ../../src/logging.cpp)
target_compile_features(router_metadata_bench PRIVATE cxx_std_17)
target_compile_definitions(router_metadata_bench PRIVATE RDMA_PROXY_HAVE_CUDA=0 RDMA_PROXY_HAVE_VERBS=0)
target_include_directories(router_metadata_bench PRIVATE ../../include)
target_link_libraries(router_metadata_bench PRIVATE Threads::Threads)
target_compile_options(router_metadata_bench PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_binary_codec native_proxy.cpp ../../src/router_routing.cpp
    ../../src/protocol.cpp ../../src/logging.cpp)
  target_compile_features(test_binary_codec PRIVATE cxx_std_17)
  target_compile_definitions(test_binary_codec PRIVATE RDMA_PROXY_HAVE_CUDA=0
    RDMA_PROXY_HAVE_VERBS=0 ROUTER_METADATA_BENCH_CODEC_TEST=1)
  target_include_directories(test_binary_codec PRIVATE ../../include)
  target_link_libraries(test_binary_codec PRIVATE Threads::Threads)
  add_test(NAME binary_codec COMMAND test_binary_codec)
  set_tests_properties(binary_codec PROPERTIES TIMEOUT 15)
endif()
