add_executable(bench_main
    main.cpp
)
target_link_libraries(bench_main PRIVATE ${PROJECT_NAME})
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
