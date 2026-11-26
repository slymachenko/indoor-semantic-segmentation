from abc import ABC, abstractmethod

from typing import Any
from numpy.typing import NDArray


class SegmentationDataset(ABC):
    """Interface for image segmentation datasets."""

    @abstractmethod
    def get_pair(self, index: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return an image and its segmentation mask."""
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def close(self) -> None: ...
