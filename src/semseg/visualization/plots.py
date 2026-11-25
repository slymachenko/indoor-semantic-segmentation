"""Plotting helpers that do not perform model inference or data loading."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from numpy.typing import ArrayLike, NDArray
from typing import Any, Sequence


def _save_figure(fig: Any, save_path: str | Path | None, dpi: int) -> None:
    if save_path is None:
        return
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")


def plot_segmentation_samples(
    images: Sequence[ArrayLike] | NDArray,
    ground_truth: Sequence[ArrayLike] | NDArray,
    predictions: Sequence[ArrayLike] | NDArray,
    probabilities: Sequence[ArrayLike] | NDArray | None = None,
    *,
    indices: Sequence[int] | None = None,
    max_samples: int = 4,
    save_path: str | Path | None = None,
    dpi: int = 160,
) -> tuple[Any, Any]:
    """Plot RGB images, ground-truth masks, predicted masks, and optional scores.

    Inputs are image/mask arrays already prepared by the caller. This function
    deliberately does not load datasets, extract features, or run inference.
    """
    image_list = list(images)
    truth_list = list(ground_truth)
    prediction_list = list(predictions)
    probability_list = None if probabilities is None else list(probabilities)
    lengths = [len(image_list), len(truth_list), len(prediction_list)]
    if probability_list is not None:
        lengths.append(len(probability_list))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Images, masks, predictions, and probabilities must align; got lengths {lengths}."
        )
    n_items = lengths[0]
    if n_items == 0:
        raise ValueError("At least one aligned image/mask/prediction is required.")
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")
    n_rows = min(n_items, max_samples)
    if indices is not None and len(indices) < n_rows:
        raise ValueError("indices must contain at least one index per plotted sample.")
    n_cols = 4 if probability_list is not None else 3

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.25 * n_rows),
        squeeze=False,
    )
    headings = ["Image", "Ground truth"]
    if probability_list is not None:
        headings.append("Foreground probability")
    headings.append("Prediction")

    for row in range(n_rows):
        image = np.asarray(image_list[row])
        truth = np.asarray(truth_list[row])
        prediction = np.asarray(prediction_list[row])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Image at row {row} must have shape (H, W, 3).")
        if truth.shape != prediction.shape or truth.shape != image.shape[:2]:
            raise ValueError(
                f"Image, ground truth, and prediction shapes differ at row {row}."
            )

        axes[row, 0].imshow(image)
        axes[row, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        if probability_list is not None:
            probability = np.asarray(probability_list[row])
            if probability.shape != truth.shape:
                raise ValueError(f"Probability map shape differs at row {row}.")
            axes[row, 2].imshow(probability, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(prediction, cmap="gray", vmin=0, vmax=1)

        sample_label = (
            f"Image {indices[row]}" if indices is not None else f"Sample {row + 1}"
        )
        axes[row, 0].set_ylabel(sample_label)
        for col in range(n_cols):
            axes[row, col].axis("off")

    for col, title in enumerate(headings):
        axes[0, col].set_title(title)

    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, axes


def plot_classifier_performance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: ArrayLike | None = None,
    *,
    class_names: Sequence[str] = ("background", "foreground"),
    save_path: str | Path | None = None,
    dpi: int = 160,
) -> tuple[Any, Any]:
    """Plot normalized confusion matrix and optional ROC/precision-recall curves."""
    truth = np.asarray(y_true).ravel()
    prediction = np.asarray(y_pred).ravel()
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")

    has_scores = y_proba is not None and np.unique(truth).size == 2
    n_cols = 3 if has_scores else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 4.6), squeeze=False)
    axis = axes[0]

    ConfusionMatrixDisplay.from_predictions(
        truth,
        prediction,
        labels=[0, 1],
        display_labels=list(class_names),
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        ax=axis[0],
        colorbar=False,
    )
    axis[0].set_title("Normalized confusion matrix")

    if has_scores:
        scores = np.asarray(y_proba).ravel()
        if scores.shape != truth.shape:
            raise ValueError("y_proba must match y_true.")
        RocCurveDisplay.from_predictions(truth, scores, ax=axis[1])
        axis[1].set_title("ROC curve")
        PrecisionRecallDisplay.from_predictions(truth, scores, ax=axis[2])
        axis[2].set_title("Precision-recall curve")

    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, axes


def plot_metric_comparison(
    runs: dict[str, dict[str, float | int | None]],
    metrics: Sequence[str] = ("iou", "dice", "precision", "recall"),
    *,
    save_path: str | Path | None = None,
    dpi: int = 160,
) -> tuple[Any, Any]:
    """Create a grouped bar chart comparing named metrics across runs."""
    run_names = list(runs)
    available = [
        metric
        for metric in metrics
        if any(runs[name].get(metric) is not None for name in run_names)
    ]
    if not run_names or not available:
        raise ValueError("No run metrics are available for comparison.")

    values = np.array(
        [
            [
                np.nan if runs[run].get(metric) is None else float(runs[run][metric])
                for metric in available
            ]
            for run in run_names
        ]
    )
    fig, ax = plt.subplots(figsize=(max(7, len(run_names) * 1.5), 4.8))
    x = np.arange(len(run_names))
    width = 0.8 / len(available)
    for j, metric in enumerate(available):
        ax.bar(
            x + (j - (len(available) - 1) / 2) * width,
            values[:, j],
            width,
            label=metric,
        )
    ax.set_xticks(x, run_names, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Segmentation metric comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, ax
