## GGB Benchmarking

This directory contains the C++ benchmark harness and the Python workload generates for the `ggb::FeatureStore`.

### Directory Structure

The suite uses a structured data hierarchy to manage raw datasets, sampled workloads, and performance results.

```bash
bench/data/
└── <dataset_name>/                   # e.g., ogbn-arxiv
    ├── raw_data.zip                  # Original downloaded archive
    ├── edge.csv                      # Extracted topology
    ├── node-feat.csv                 # Extracted features
    └── <run_id>/                     # e.g., run-0001
        ├── metadata.json             # Sampling params (fan-out, hops, etc.)
        ├── queries.csv               # Sampled mini-batch Node IDs
        └── results/                  # Performance telemetry
            └── result_<engine>_<time>.json
```


### Workload Generation

The benchmark requires pre-generated query workloads (mini-batch calls to `ggb::FeatureStore.get_multi_tensor_async`). These are managed by the [query_gen](./query_gen/) package.

1. **Setup Environment**: Initializes the Python virtual environment (requires `uv`) and installs dependencies (e.g. `pyg-lib`):

```bash
../scripts/bench_setup_python.sh
```

2. **Generate Workloads**: Create mini-batch query requests via offline sampling with `pyg-lib`. Specify the `ogbn-` dataset and sampling parameters:

```bash
../scripts/bench_run_query_gen.sh --dataset-name ogbn-arxiv --num-hops 2 --fan-out 10
```

- *Extraction*: Automatically extracts `edge.csv` and `node-feat.csv` from the raw data
- *Versioning*: Outputs sampled batches to a new `run-id` directory (e.g. `run-0001`). The ID auto-increments across invocation.

### C++ Performance Harness

The C++ runner measures latency and throughput for different feature store engines.

#### Quick Start

The simplest way to build and run a benchmark:

```bash
../scripts/bench_run.sh ogbn-arxiv run-0001
```

This script will automatically compile `ggb` and the benchmark harness under *release* mode, then execute the binary.

By default, this will run all the engines. You can also run a specific engine like:

```bash
../scripts/bench_run.sh ogbn-arxiv run-0001 --engine mmap
```

Alternatively, you can manually compile and execute:

```bash
../scripts/ggb_build.sh Release --bench # Compile
../build/bench/bench_main # Execute
```

#### Results and Reporting

After execution, metrics are logged to the console and saved as JSOn file in the corresponding `results/` directory.
You can create your own sinks to process the benchmarking records.

**Key Metrics Captures**:
- Latency: mean, std, p50, p95, p99
- Throughput: qps, tensors-per-second, memory bandwith
- Provenance: git-hash, timestamp info
