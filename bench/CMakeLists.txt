include(FetchContent)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_MakeAvailable(json)

add_executable(bench_main
    main.cpp
)
target_link_libraries(bench_main PRIVATE
    ${PROJECT_NAME}
    nlohmann_json::nlohmann_json
)
target_include_directories(bench_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_include_directories(bench_main PRIVATE
    # 1. Look in 'src' so #include "common/logging.h" works
    ${CMAKE_CURRENT_SOURCE_DIR}/../src

    # 2. Look in project root so #include "bench/common/xxx.h" works
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_definitions(-DBENCHMARKS_ROOT="${CMAKE_SOURCE_DIR}")

execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions(-DGGB_GIT_HASH="${GIT_HASH}")
