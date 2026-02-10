## GGB

High-performance Feature stores for out-of-core GNN training workloads.

### Prerequisites

- Clang 18+
- LLVM libc++
- CMake 3.16+
- Python 3.10+ (for query generation)

### Development

#### Build

The project is tested on `Clang` with `libc++`. Use the provided build script to configure and compile:

```bash
./scripts/ggb_build.sh [Debug|Release|RelWithDebInfo] [--bench|--no-bench]
```

#### Testing

Run the test suite using:

```bash
./scripts/ggb_run_tests.sh
```

#### Benchmarks

Refer to [bench/](./bench/).
