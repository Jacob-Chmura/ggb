#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BENCH_DIR="${PROJECT_ROOT}/bench"
PYTHON_ROOT_DIR="${BENCH_DIR}/query_gen"
VENV_DIR="$PYTHON_ROOT_DIR/.venv"

print_usage() {
    echo "Usage: $0 [PYTHON ARGS]"
    echo
    echo "Run the Python query generation benchmark from bench/query_gen subdirectory."
    echo "All arguments are forwarded to bench/query_gen/__main__.py."
    echo
    echo "Example:"
    echo "  $0 --dataset-name ogbn-arxiv --batch-size 512 --num-hops 2 --fan-out 10"
    echo
    echo "Environment:"
    echo "  - Checks for 'uv' installation."
    echo "  - Activates the Python virtual environment in $VENV_DIR."
    echo "  - Assumes you already ran setup_bench_python.sh."
}

check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Python virtual environment not found in $VENV_DIR."
        echo "Please run setup_bench_python.sh first."
        exit 1
    fi
}

check_uv_install() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
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

    check_venv
    check_uv_install

    cd "$PYTHON_ROOT_DIR"
    uv run main.py "$@"
}

main "$@"
