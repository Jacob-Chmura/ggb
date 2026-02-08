add_executable(bench_main
    main.cpp
)
target_link_libraries(bench_main PRIVATE ${PROJECT_NAME})
target_include_directories(bench_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_definitions(-DBENCHMARKS_ROOT="${CMAKE_SOURCE_DIR}")
