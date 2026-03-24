from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_split(
    df: pd.DataFrame,
    label_column: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + val_size),
        stratify=df[label_column],
        random_state=seed,
    )
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val_size),
        stratify=temp_df[label_column],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def write_split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_csv: str | Path,
) -> None:
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)

