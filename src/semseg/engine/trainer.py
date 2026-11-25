from semseg.evaluation.metrics import evaluate_binary_segmentation

from semseg.models.base import SegmentationModel
from numpy.typing import ArrayLike
from typing import Any


def train_model(
    model: SegmentationModel,
    train_features: ArrayLike,
    train_targets: ArrayLike,
    validation_features: ArrayLike,
    validation_targets: ArrayLike,
    *,
    threshold: float = 0.5,
) -> tuple[SegmentationModel, dict[str, Any]]:
    """Fit a model and evaluate it on validation data."""
    model.fit(train_features, train_targets)
    probabilities = model.predict_proba(validation_features)
    predictions = model.predict(validation_features, threshold=threshold)
    metrics = evaluate_binary_segmentation(
        validation_targets, predictions, probabilities
    )
    return model, metrics
