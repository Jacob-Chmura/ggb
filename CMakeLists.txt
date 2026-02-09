cmake_minimum_required(VERSION 3.16)
project(
    ggb
    VERSION 0.1.0
    DESCRIPTION "Feature stores for out-of-core GNN training workloads"
    LANGUAGES CXX
)
option(GGB_BUILD_BENCHMARKS "Build benchmark binaries" OFF)
option(GGB_BUILD_TESTS "Build test binaries" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

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


if (GGB_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

if (GGB_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(test_ggb
        test/test_engine_factory.cpp
        test/test_feature_store.cpp
        test/test_io.cpp
        test/test_mmap_region.cpp
    )
    target_link_libraries(test_ggb PRIVATE
        ${PROJECT_NAME}
        GTest::gtest_main
    )
    target_include_directories(test_ggb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    include(GoogleTest)
    gtest_discover_tests(test_ggb)
endif()
