## GGB Benchmarking

This directory contains the C++ benchmark suite and the Python tools for generating query workloads.

### C++ Benchmarks

To compile the benchmarks, use the `--bench` flag with the main build script:

```bash
../scripts/ggb_build.sh Release --bench
```

The resulting binaries will be located in `build/bench/`.

### Query Workloads

The suite requires generated query workloads (mini-batch calls to `ggb::FeatureStore.get_multi_tensor_async`).
These are managed by the [query_gen](./query_gen/) package.

1. **Setup Environment**: Initializes the Python virtual environment (requires `uv`) and installs dependencies (e.g. `pyg-lib`):

```bash
../scripts/bench_setup_python.sh
```

2. **Generate Workloads**: Create mini-batch query requests via offline sampling with `pyg-lib`. Specify the `ogbn-` dataset and sampling parameters:

```bash
../scripts/bench_run_query_gen.sh --dataset-name ogbn-arxiv --num-hops 2 --fan-out 10
```

This will download the raw `ogbn-arxiv` tarbell and output sampled batches to a timestamped directory (`data/`, by default).

3. **Execution**: Run the various feature stores on the dataset:

```bash
../build/bench/bench_main
```
