import argparse
import gzip
import pathlib
import shutil
import subprocess
import zipfile

parser = argparse.ArgumentParser(
    description="Download and preprocess an OGBN dataset.",
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

_OGB_NODE_PROP_BASE_DOWNLOAD_URL = "http://snap.stanford.edu/ogb/data/nodeproppred"


def main() -> None:
    args = parser.parse_args()
    download_and_extract(dataset_name=args.dataset_name, dataset_dir=args.dataset_dir)


def download_and_extract(
    dataset_name: str, dataset_dir: str
) -> dict[str, pathlib.Path]:
    save_path = _download_data(dataset_name=dataset_name, dataset_dir=dataset_dir)
    extracted_files = _extract_data(save_path)
    return extracted_files


def _download_data(dataset_name: str, dataset_dir: str) -> pathlib.Path:
    raw_file_name = _dataset_name_to_raw_file_name(dataset_name)
    save_path = pathlib.Path(dataset_dir) / dataset_name / raw_file_name
    if save_path.exists():
        print(f"'{save_path}' already exists, skipping download")
    else:
        url = f"{_OGB_NODE_PROP_BASE_DOWNLOAD_URL}/{raw_file_name}"
        print(f"Downloading '{dataset_name}' from '{url}' to '{save_path}'")
        subprocess.run(["wget", "-P", str(save_path.parent), url], check=True)
    return save_path


def _extract_data(zip_path: pathlib.Path) -> dict[str, pathlib.Path]:
    files_to_extract = {
        "edgelist": "raw/edge.csv.gz",
        "node_features": "raw/node-feat.csv.gz",
    }

    # zip_path = .../ogbn-arxiv/arxiv.zip
    save_dir = zip_path.parent  # extract next to the zip

    extracted_files = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for file_type, file_name in files_to_extract.items():
            member = f"{zip_path.stem}/{file_name}"  # arxiv/raw/...
            print(f"Extracting {file_type} from '{member}'...", end="")

            # Decompress gzip into csv
            extracted_gz = pathlib.Path(zf.extract(member, path=save_dir))
            final_path = save_dir / extracted_gz.stem
            with gzip.open(extracted_gz, "rb") as f_in, open(final_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            # cleanup empty directories
            extracted_gz.unlink()
            extracted_gz.parent.rmdir()
            extracted_gz.parent.parent.rmdir()
            print("OK")

            extracted_files[file_type] = final_path
    return extracted_files


def _dataset_name_to_raw_file_name(dataset_name: str) -> str:
    return f"{dataset_name.split('-')[1]}.zip"  # ogbn-arxiv -> arxiv.zip


if __name__ == "__main__":
    main()
