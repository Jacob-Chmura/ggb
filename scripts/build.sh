#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_TYPE="${1:-RelWithDebInfo}"  # default if not provided

main() {
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    echo "Configuring CMake project (Build type: $BUILD_TYPE)..."
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -S "$PROJECT_ROOT" -B "$BUILD_DIR"

    echo "Building project..."
    cmake --build . -- "-j$(nproc)"

    # Copy compile_commands.json to project root for clang-tidy
    cp compile_commands.json "$PROJECT_ROOT/"
    echo "Build complete. compile_commands.json copied to project root."
}

main "$@"
