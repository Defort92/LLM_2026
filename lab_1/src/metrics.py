from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
        }


def calculate_metrics(
    targets: list[int],
    predictions: list[int],
) -> tuple[ClassificationMetrics, np.ndarray]:
    metrics = ClassificationMetrics(
        accuracy=accuracy_score(targets, predictions),
        precision_macro=precision_score(
            targets, predictions, average="macro", zero_division=0
        ),
        recall_macro=recall_score(
            targets, predictions, average="macro", zero_division=0
        ),
        f1_macro=f1_score(targets, predictions, average="macro", zero_division=0),
    )
    matrix = confusion_matrix(targets, predictions)
    return metrics, matrix

