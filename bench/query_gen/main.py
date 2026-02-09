import argparse

from download_ogbn_data import download_and_extract
from generate_queries import create_run_dir, generate_queries

parser = argparse.ArgumentParser(
    description="Generate graph neighborhood query workloads for benchmarking feature stores.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset-name",
    type=str,
    default="ogbn-arxiv",
    choices=["ogbn-arxiv", "ogbn-products"],
    help="Dataset name.",
)
parser.add_argument(
    "--dataset-dir", type=str, default="../data", help="Root directory of the dataset."
)
parser.add_argument("--seed", type=int, default=0, help="RNG for reproducibility.")
parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
parser.add_argument("--num-hops", type=int, default=2, help="Number of hops to sample.")
parser.add_argument(
    "--fan-out", type=int, default=10, help="Number of neighbors to sample at each hop."
)


def main():
    args = parser.parse_args()

    extracted_files = download_and_extract(
        dataset_name=args.dataset_name, dataset_dir=args.dataset_dir
    )
    edgelist_file = extracted_files["edgelist"]
    output_root = edgelist_file.parent
    run_dir = create_run_dir(output_root=output_root, metadata=vars(args))

    generate_queries(
        edgelist_file_str=str(edgelist_file),
        seed=args.seed,
        batch_size=args.batch_size,
        num_hops=args.num_hops,
        fan_out=args.fan_out,
        output_dir=run_dir,
    )


if __name__ == "__main__":
    main()
