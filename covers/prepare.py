"""Extract the dataset zip archives to disk.

Usage:
    python -m covers.prepare --dataset_path dataset/dataset
"""
from __future__ import annotations

import argparse
import os
import zipfile

from covers import data as data_mod


def extract(zip_path: str, dest_dir: str) -> None:
    print(f"extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print("done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract train/test zips")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--train_zip", default=None)
    parser.add_argument("--test_zip", default=None)
    args = parser.parse_args()

    dataset_path = data_mod.resolve_data_path(args.dataset_path)
    train_zip = args.train_zip or os.path.join(dataset_path, "train.zip")
    test_zip = args.test_zip or os.path.join(dataset_path, "test.zip")

    if os.path.isdir(os.path.join(dataset_path, "train")) and not args.train_zip:
        print("train/ already extracted, skipping")
    elif os.path.exists(train_zip):
        extract(train_zip, dataset_path)
    else:
        print(f"train.zip not found: {train_zip}")

    if os.path.isdir(os.path.join(dataset_path, "test")) and not args.test_zip:
        print("test/ already extracted, skipping")
    elif os.path.exists(test_zip):
        extract(test_zip, dataset_path)
    else:
        print(f"test.zip not found: {test_zip}")


if __name__ == "__main__":
    main()
