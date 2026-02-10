## GGB

High-performance Feature stores for out-of-core GNN training workloads.

### Prerequisites

- Clang 18+
- LLVM libc++
- CMake 3.10+
- Python 3.10+ (for benchmarks and query generation)

### Development

#### Build

The project uses Clang with `libc++` by default. Use the provided build script to configure and compile:

```bash
./scripts/ggb_build.sh [Debug|Release|RelWithDebInfo] [--bench|--no-bench]
```

#### Testing

Run the test suite via the script (requires `ctest`):

```bash
./scripts/ggb_run_tests.sh
```
