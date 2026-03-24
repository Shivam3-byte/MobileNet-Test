from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_from_manifest(manifest_csv: str, output_root: str) -> None:
    df = pd.read_csv(manifest_csv)
    required_columns = {"image_path", "label", "split"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for row in df.itertuples(index=False):
        source = Path(row.image_path)
        destination = Path(output_root) / row.split / str(row.label) / source.name
        copy_file(source, destination)


def build_from_imagefolder(raw_root: str, output_root: str, train_ratio: float = 0.7, val_ratio: float = 0.15) -> None:
    raw_path = Path(raw_root)
    class_dirs = [item for item in raw_path.iterdir() if item.is_dir()]
    for class_dir in class_dirs:
        images = sorted([item for item in class_dir.iterdir() if item.is_file()])
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }
        for split_name, split_images in splits.items():
            for image_path in split_images:
                destination = Path(output_root) / split_name / class_dir.name / image_path.name
                copy_file(image_path, destination)


def main():
    parser = argparse.ArgumentParser(description="Create ImageFolder-compatible dataset splits.")
    parser.add_argument("--mode", choices=["manifest", "imagefolder"], required=True)
    parser.add_argument("--input", required=True, help="Input CSV manifest or raw ImageFolder root")
    parser.add_argument("--output", required=True, help="Output root for processed data")
    args = parser.parse_args()

    if args.mode == "manifest":
        build_from_manifest(args.input, args.output)
    else:
        build_from_imagefolder(args.input, args.output)

    print(f"Prepared data at {args.output}")


if __name__ == "__main__":
    main()

