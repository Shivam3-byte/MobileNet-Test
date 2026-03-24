from __future__ import annotations


def compute_severity_percent(lesion_pixels: int, leaf_pixels: int) -> float:
    if leaf_pixels <= 0:
        return 0.0
    return (lesion_pixels / leaf_pixels) * 100.0


def severity_label(severity_percent: float) -> str:
    if severity_percent < 10:
        return "Mild"
    if severity_percent < 25:
        return "Moderate"
    return "Severe"

