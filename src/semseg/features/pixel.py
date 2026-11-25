import numpy as np

from semseg.data.preprocessing import binarize_mask

from numpy.typing import ArrayLike, NDArray
from collections.abc import Sequence


def image_features(
    image: ArrayLike,
    with_position: bool = False,
) -> NDArray[np.float32]:
    """Extract RGB and optionally spatial features for each pixel."""
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an RGB image (H, W, 3).")

    height, width = image.shape[:2]
    features = image.reshape(-1, 3).astype(np.float32) / 255.0

    if with_position:
        x = np.linspace(0, 1, width, dtype=np.float32)
        y = np.linspace(0, 1, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack((xx.ravel(), yy.ravel()))
        features = np.column_stack((features, positions)).astype(np.float32)

    return features


def image_to_pixels(
    image: ArrayLike,
    labels: ArrayLike,
    target_class_ids: Sequence[int],
    with_position: bool = False,
    ignore_index: int | None = 0,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Create a supervised pixel-level dataset."""
    image_arr = np.asarray(image)
    labels_arr = np.asarray(labels)

    if image_arr.shape[:2] != labels_arr.shape:
        raise ValueError("labels must match image height and width.")

    flat_labels = labels_arr.ravel()
    valid = np.ones(flat_labels.shape, dtype=bool)
    if ignore_index is not None:
        valid &= flat_labels != ignore_index

    features = image_features(image_arr, with_position)[valid]
    targets = binarize_mask(flat_labels[valid], target_class_ids)
    return features, targets


def build_pixel_dataset(
    dataset,
    indices: Sequence[int],
    target_class_ids: Sequence[int],
    *,
    stride: int = 1,
    with_position: bool = False,
    ignore_index: int | None = 0,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Aggregate pixel-level samples from selected images."""

    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    if len(indices) == 0:
        raise ValueError("indices cannot be empty.")

    X_parts = []
    y_parts = []

    for index in indices:
        image, labels = dataset.get_pair(index)

        image = image[::stride, ::stride]
        labels = labels[::stride, ::stride]

        X, y = image_to_pixels(
            image,
            labels,
            target_class_ids,
            with_position=with_position,
            ignore_index=ignore_index,
        )

        if len(y) > 0:
            X_parts.append(X)
            y_parts.append(y)

    if not X_parts:
        raise ValueError("No valid labeled pixels were found in the selected images.")

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
    )
