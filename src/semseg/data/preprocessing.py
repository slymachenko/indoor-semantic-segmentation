import numpy as np

from numpy.typing import ArrayLike, NDArray


def binarize_mask(labels: ArrayLike, target_class_ids: ArrayLike) -> NDArray[np.uint8]:
    """Convert target class identifiers in a label mask to binary labels"""
    return np.isin(labels, target_class_ids).astype(np.uint8)
