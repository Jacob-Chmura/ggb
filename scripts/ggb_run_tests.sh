#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_ROOT/build"

BUILD_TYPE="Debug"
EXTRA_CTEST_ARGS=""

print_usage() {
    echo "Usage: $0 [ctest-args]"
    echo
    echo "Arguments:"
    echo "  ctest-args    Arguments passed directly to ctest (e.g. -R InMemory)"
}

parse_args() {
    for arg in "$@"; do
        case "$arg" in
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                # Treat any other arguments as ctest filters/args
                EXTRA_CTEST_ARGS+="$arg "
                ;;
        esac
    done
}

configure_cmake() {
    echo "Configuring CMake project (Build type: $BUILD_TYPE) in $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DGGB_BUILD_TESTS=ON \
          -DGGB_BUILD_BENCHMARKS=OFF \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          "$PROJECT_ROOT"
}

build_project() {
    cmake --build . -- "-j$(nproc)"
}

run_tests() {
    ctest --output-on-failure $EXTRA_CTEST_ARGS
}


main() {
    parse_args "$@"
    configure_cmake
    build_project
    run_tests
}

main "$@"
