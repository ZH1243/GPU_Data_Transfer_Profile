cmake_minimum_required(VERSION 3.16)

project(nvlink_iteration_done_benchmark LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(iteration_done_benchmark src/main.cpp)
target_compile_features(iteration_done_benchmark PRIVATE cxx_std_17)
target_compile_options(iteration_done_benchmark PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>)
target_link_libraries(iteration_done_benchmark PRIVATE Threads::Threads)

if(UNIX AND NOT APPLE)
    target_link_libraries(iteration_done_benchmark PRIVATE rt)
endif()

enable_testing()
add_test(
    NAME iteration_done_smoke
    COMMAND iteration_done_benchmark
            --proxies=2
            --warmup=100
            --iterations=1000
            --queue-depth=64)
