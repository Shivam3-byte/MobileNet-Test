from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    confusion: list[list[int]]
    report: dict


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> ClassificationMetrics:
    confusion = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=list(class_names), output_dict=True, zero_division=0)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        confusion=confusion,
        report=report,
    )


def pretty_print_metrics(metrics: ClassificationMetrics) -> str:
    return (
        f"Accuracy: {metrics.accuracy:.4f}\n"
        f"Macro F1: {metrics.macro_f1:.4f}\n"
        f"Confusion Matrix: {np.array(metrics.confusion)}"
    )

