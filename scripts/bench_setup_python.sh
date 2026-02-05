#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
PYTHON_ROOT_DIR="${PROJECT_ROOT}/bench/query_gen"

# Needs to be manually patched into the .venv to avoid build conflicts with torch
PYG_LIB_URL="https://data.pyg.org/whl/torch-2.8.0+cpu.html"

print_usage() {
    echo "Usage: $0"
    echo
    echo "Setup Python environment for benchmarks."
    echo
    echo "Environment:"
    echo "  Checks for 'uv' installation."
    echo "  Creates .venv in bench/query_gen if missing."
    echo "  Runs 'uv sync' in the bench/query_gen directory."
}

check_uv_install() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

parse_args() {
    for arg in "$@"; do
        case $arg in
            --help|-h)
                print_usage
                exit 0
                ;;
        esac
    done
}

setup_base_venv() {
    echo "Setting up Python environment in $PYTHON_ROOT_DIR..."
    cd "$PYTHON_ROOT_DIR"
    uv sync
}

add_pyg_lib() {
    # We use `uv pip install` instead of `uv add` since otherwise,
    # future calls to `uv sync` result in build resolution problems
    # with torch and pyg-lib.
    echo "Adding pyg-lib wheel from $PYG_LIB_URL..."
    uv pip install pyg-lib -f "$PYG_LIB_URL"
}

main() {
    parse_args "$@"
    check_uv_install
    setup_base_venv
    add_pyg_lib
    echo "Python environment ready in $PYTHON_ROOT_DIR/.venv"
}

main "$@"
