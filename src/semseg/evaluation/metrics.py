from typing import Any

from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    jaccard_score,
    confusion_matrix,
)


def evaluate_binary_segmentation(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: ArrayLike,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "ap": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "iou": jaccard_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=target_names),
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    for name in ("auc", "average_precision", "accuracy", "precision", "recall", "iou"):
        if name in metrics:
            formatted = "n/a" if metrics[name] is None else f"{metrics[name]:.3f}"
            print(f"{name.replace('_', ' ').title():<20} {formatted}")

    if "confusion_matrix" in metrics:
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])
