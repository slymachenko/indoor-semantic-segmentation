"""Small common interface implemented by segmentation models."""

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray


class SegmentationModel(ABC):
    """Interface expected by the training and inference engines.

    Models consume feature matrices (one row per pixel/patch) and return
    binary labels and foreground probabilities.
    """

    @abstractmethod
    def fit(self, features: ArrayLike, targets: ArrayLike) -> "SegmentationModel": ...

    @abstractmethod
    def predict(self, features: ArrayLike, threshold: float = 0.5) -> NDArray: ...

    @abstractmethod
    def predict_proba(self, features: ArrayLike) -> NDArray: ...
