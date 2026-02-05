#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_ROOT/build"

BUILD_TYPE="RelWithDebInfo"
BUILD_BENCHMARKS="OFF"

print_usage() {
    echo "Usage: $0 [BUILD_TYPE] [--bench|--no-bench]"
    echo
    echo "Arguments:"
    echo "  BUILD_TYPE    Debug | Release | RelWithDebInfo | MinSizeRel"
    echo "                Default: RelWithDebInfo"
    echo
    echo "Options:"
    echo "  --bench       Build benchmarks"
    echo "  --no-bench    Do not build benchmarks (default)"
}

parse_args() {
    for arg in "$@"; do
        case "$arg" in
            Debug|Release|RelWithDebInfo|MinSizeRel)
                BUILD_TYPE="$arg"
                ;;
            --bench)
                BUILD_BENCHMARKS="ON"
                ;;
            --no-bench)
                BUILD_BENCHMARKS="OFF"
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                echo "Error: Unknown argument '$arg'"
                print_usage
                exit 1
                ;;
        esac
    done
}

clean_build_dir() {
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
}

configure_cmake() {
    echo "Configuring CMake project (Build type: $BUILD_TYPE)..."
    cd "$BUILD_DIR"
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DGGB_BUILD_BENCHMARKS="$BUILD_BENCHMARKS" \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          "$PROJECT_ROOT"
}

build_project() {
    echo "Building project..."
    cmake --build . -- "-j$(nproc)"
}

copy_compile_commands() {
    cp compile_commands.json "$PROJECT_ROOT/"
}

main() {
    parse_args "$@"
    clean_build_dir
    configure_cmake
    build_project
    copy_compile_commands
}

main "$@"
