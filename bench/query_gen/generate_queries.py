import argparse
import csv
import json
import pathlib
from datetime import datetime, timezone

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Generate feature store queries from an edgelist file.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--edgelist-file", type=str, required=True, help="Path to edgelist csv file."
)
parser.add_argument(
    "--rounds", type=int, default=1, help="Number of epochs to simulate."
)
parser.add_argument("--base-seed", type=int, default=0, help="RNG for reproducibility.")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
parser.add_argument("--num_hops", type=int, default=2, help="Number of hops to sample.")
parser.add_argument(
    "--fan-out", type=int, default=10, help="Number of neighbors to sample at each hop."
)


def main() -> None:
    args = parser.parse_args()

    output_root = pathlib.Path(args.edgelist_file).parent / "queries"
    run_dir = create_run_dir(output_root=output_root, metadata=vars(args))

    for i in range(args.rounds):
        seed = args.base_seed + i
        generate_queries(
            edgelist_file_str=args.edgelist_file,
            seed=seed,
            batch_size=args.batch_size,
            num_hops=args.num_hops,
            fan_out=args.fan_out,
            output_dir=run_dir,
        )


def generate_queries(
    edgelist_file_str: str,
    seed: int,
    batch_size: int,
    num_hops: int,
    fan_out: int,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    print(f"Generating queries (seed={seed})")
    seed_everything(seed)

    edge_index = _read_edgelist(edgelist_file_str)
    loader = NeighborLoader(
        data=Data(edge_index=edge_index, num_nodes=1 + edge_index.max()),
        num_neighbors=[fan_out] * num_hops,
        batch_size=batch_size,
        shuffle=True,
    )

    output_file = output_dir / f"queries-{seed}.csv"
    print(f"Saving queries to '{output_file}'")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        for batch in tqdm(loader):
            writer.writerow(batch.n_id.tolist())

    return output_file


def create_run_dir(output_root: pathlib.Path, metadata: dict) -> pathlib.Path:
    print(f"Creating run sub-directory at '{output_root}'")
    output_root.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("run-")
    )
    run_id = len(existing) + 1
    run_dir = output_root / f"run-{run_id:04d}"
    run_dir.mkdir()
    print(f"Run sub-directory created at: {run_dir}")

    metadata["created_at"] = datetime.now(timezone.utc).isoformat()
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=4)
    return run_dir


def _read_edgelist(edgelist_file_str: str) -> torch.LongTensor:
    edgelist_file = pathlib.Path(edgelist_file_str)
    if not edgelist_file.exists():
        raise FileNotFoundError(f"Edgelist file '{edgelist_file}' does not exist")

    print(f"Reading edgelist from '{edgelist_file}'")
    with open(edgelist_file, mode="r") as f:
        reader = csv.reader(f)
        edges = [[int(src), int(dst)] for src, dst in reader]
    print(f"Read {len(edges)} edges")

    edge_index = torch.LongTensor(edges).T.contiguous()
    if not edge_index.numel():
        raise ValueError("Got an empty edge list")
    return edge_index


if __name__ == "__main__":
    main()
