from __future__ import annotations

from pathlib import Path


def create_yolo_seg_structure(root_dir: str):
    root = Path(root_dir)
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

