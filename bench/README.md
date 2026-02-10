## GGB Benchmarking

This directory contains the C++ benchmark suite and the Python-based query generation toolset.

### C++ Benchmarks

To build the benchmarks, ensure the `--bench` flag is used during the main build process:

```bash
../scripts/ggb_build.sh Release --bench
```

The resulting binary will be located in `build/bench/` directory.

### Query Workloads

The benchmarking suite depend son generated query workloads. These are managed by the [query_gen](./query_gen/) Python package.

To generate a workload, first setup your python environment:

```bash
../scripts/bench_setup_python.sh
```

This will build the `.venv` with the required dependencies to generate realistic graph samples (e.g. with `pyg-lib`).

Now we can generate a workload by specifying the dataset and sampling parameters:

```bash
../scripts/bench_run_query_gen.sh --dataset-name ogbn-arxiv --num-hops 2 --fan-out 10
```

which forward the arguments to the [python entry point](./query_gen/main.py).

This script will download the required `ogbn-arxiv` raw tarbell, extract the edgelist, and run `pyg-lib`
neighbor sampling to extract query batches to a timestamped run directory within the data path.
