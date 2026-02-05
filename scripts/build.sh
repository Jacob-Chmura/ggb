#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_ROOT/build"
BUILD_TYPE="RelWithDebInfo"  # default build type

print_usage() {
    echo "Usage: $0 [BUILD_TYPE]"
    echo
    echo "Build the C++ project using CMake."
    echo
    echo "Arguments:"
    echo "  BUILD_TYPE    Optional. CMake build type (Debug, Release, RelWithDebInfo, MinSizeRel)."
    echo "                Default: RelWithDebInfo"
    echo
    echo "Environment:"
    echo "  - Generates compile_commands.json for clang-tidy."
    echo "  - Builds all targets in the build directory."
}

parse_args() {
    if [ $# -gt 1 ]; then
        echo "Error: Too many arguments."
        print_usage
        exit 1
    fi

    if [ $# -eq 1 ]; then
        case "$1" in
            Debug|Release|RelWithDebInfo|MinSizeRel)
                BUILD_TYPE="$1"
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                echo "Error: Invalid build type '$1'."
                print_usage
                exit 1
                ;;
        esac
    fi
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
    echo "compile_commands.json copied to project root."
}

main() {
    parse_args "$@"
    clean_build_dir
    configure_cmake
    build_project
    copy_compile_commands
    echo "Build complete."
}

main "$@"
