"""XGBoost segmentation model."""

import numpy as np
from xgboost import XGBClassifier

from semseg.models.base import SegmentationModel

from numpy.typing import ArrayLike, NDArray
from typing import Any


class XGBoostSegmentationModel(SegmentationModel):
    def __init__(self, **params: Any):
        params.setdefault("eval_metric", "logloss")
        params.setdefault("n_jobs", -1)
        self.estimator = XGBClassifier(**params)

    def fit(
        self, features: ArrayLike, targets: ArrayLike
    ) -> "XGBoostSegmentationModel":
        self.estimator.fit(features, targets)
        return self

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float32]:
        probabilities = self.estimator.predict_proba(features)
        classes = list(self.estimator.classes_)
        positive_column = classes.index(1)
        return np.asarray(probabilities[:, positive_column], dtype=np.float32)

    def predict(self, features: ArrayLike, threshold: float = 0.5) -> NDArray[np.uint8]:
        return (self.predict_proba(features) >= threshold).astype(np.uint8)
