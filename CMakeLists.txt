cmake_minimum_required(VERSION 3.8)
project(include CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3")

add_subdirectory(extern/googletest/)
add_subdirectory(extern/pybind11)

# SIMD

include(GoogleTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(python)

add_executable(udnn_bench udnn_bench.cc)
target_link_libraries(udnn_bench LINK_PUBLIC udnn)
