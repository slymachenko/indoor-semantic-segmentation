from semseg.models.base import SegmentationModel

from typing import Any


def build_model(name: str, params: dict[str, Any] | None = None) -> SegmentationModel:
    """Instantiate a model based on its string name and parameters."""
    params = dict(params or {})
    estimator = None

    match name.lower():
        case "logistic_regression":
            from semseg.models.logreg import LogisticRegressionSegmentationModel

            estimator = LogisticRegressionSegmentationModel(**params)
        case _:
            raise ValueError(f"Unsupported model: {name}")

    return estimator
