from __future__ import annotations

import argparse

import pandas as pd

from src.utils.split_data import create_stratified_split, write_split_manifest


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test split manifest for field images.")
    parser.add_argument("--input", required=True, help="CSV with columns: image_path,label")
    parser.add_argument("--output", required=True, help="Output split manifest CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required_columns = {"image_path", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train_df, val_df, test_df = create_stratified_split(df)
    write_split_manifest(train_df, val_df, test_df, args.output)
    print(f"Saved split manifest to {args.output}")


if __name__ == "__main__":
    main()

