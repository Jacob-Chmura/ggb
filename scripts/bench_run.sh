#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_ROOT/build"
BIN="$BUILD_DIR/bench/bench_main"

print_usage() {
    echo "Usage: $0 <dataset> <run_id> [options]"
    echo
    echo "Arguments:"
    echo "  dataset      Name of the dataset (e.g., ogbn-arxiv)"
    echo "  run_id       ID of the query run (e.g., run-0001)"
    echo
    echo "Options:"
    echo "  --engine     mmap | in_memory | all (default: all)"
    echo "  --help       Show this message"

    echo "Environment:"
    echo "  - Triggers a `Release --bench` build of ggb."
    echo "  - Assumes you ran `setup_bench_python.sh` with the corresponding data parameters."
}

main() {
    for arg in "$@"; do
        case $arg in
            --help|-h)
                print_usage
                exit 0
                ;;
        esac
    done

    "$SCRIPT_DIR/ggb_build.sh" Release --bench
    exec "$BIN" "$@"
}

main "$@"
