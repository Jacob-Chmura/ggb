cmake_minimum_required(VERSION 3.16)
project(
    ggb
    VERSION 0.1.0
    DESCRIPTION "Out-of-core gather implementation, specialized for GNN training workloads"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -pedantic)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_executable(
        test_ggb
        test/test_ggb.cc
)
target_link_libraries(test_ggb PRIVATE
        ${PROJECT_NAME}
)
