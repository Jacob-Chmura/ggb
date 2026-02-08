cmake_minimum_required(VERSION 3.16)
project(
    ggb
    VERSION 0.1.0
    DESCRIPTION "Feature store for out-of-core GNN training workloads"
    LANGUAGES CXX
)
option(GGB_BUILD_BENCHMARKS "Build benchmark binaries" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -pedantic)

add_library(${PROJECT_NAME} SHARED
    src/engine_factory.cpp
    src/engines/flat_mmap/flat_mmap.cpp
    src/engines/in_memory/in_memory.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE GGB_COMPILE_LIBRARY)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(test_ggb test/test_ggb.cpp)
target_link_libraries(test_ggb PRIVATE ${PROJECT_NAME})
target_include_directories(test_ggb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (GGB_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()
