from semseg.evaluation.metrics import evaluate_binary_segmentation

from semseg.models.base import SegmentationModel
from numpy.typing import ArrayLike
from typing import Any


def predict(model: SegmentationModel, features: ArrayLike, *, threshold: float = 0.5):
    probabilities = model.predict_proba(features)
    predictions = model.predict(features, threshold=threshold)
    return predictions, probabilities


def evaluate(
    model: SegmentationModel,
    features: ArrayLike,
    targets: ArrayLike,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    predictions, probabilities = predict(model, features, threshold=threshold)
    return evaluate_binary_segmentation(targets, predictions, probabilities)
