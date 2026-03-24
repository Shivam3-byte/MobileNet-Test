from __future__ import annotations

from src.classification.infer import predict_image
from src.severity.severity_score import compute_severity_percent, severity_label


def run_full_pipeline(checkpoint_path: str, image_path: str, lesion_pixels: int | None = None, leaf_pixels: int | None = None):
    result = predict_image(checkpoint_path, image_path)
    if lesion_pixels is not None and leaf_pixels is not None:
        severity_percent = compute_severity_percent(lesion_pixels, leaf_pixels)
        result["severity_percent"] = severity_percent
        result["severity_label"] = severity_label(severity_percent)
    return result

