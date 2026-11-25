"""Patch extraction utilities for classical segmentation models."""

import numpy as np

from semseg.data.preprocessing import binarize_mask

from collections.abc import Iterator, Sequence
from numpy.typing import ArrayLike, NDArray


def _validate_patch_inputs(image: NDArray, patch_size: int, step: int) -> None:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape (height, width, 3).")
    if (
        not isinstance(patch_size, int)
        or isinstance(patch_size, bool)
        or patch_size < 1
        or patch_size % 2 == 0
    ):
        raise ValueError("patch_size must be a positive odd integer.")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer.")


def extract_patch(
    padded_image: ArrayLike,
    row: int,
    col: int,
    patch_size: int,
) -> NDArray:
    """Extract a square patch centered at (row, col) in a padded image."""
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")
    radius = patch_size // 2
    patch = np.asarray(padded_image)[
        row - radius : row + radius + 1,
        col - radius : col + radius + 1,
    ]
    if patch.shape[:2] != (patch_size, patch_size):
        raise ValueError("The supplied image is not padded enough for this patch.")
    return patch


def iter_image_patch_batches(
    image: ArrayLike,
    patch_size: int,
    *,
    step: int = 1,
    batch_size: int = 1024,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.float32]]]:
    """Yield flattened, normalized patch features for every sampled pixel.

    Each yielded pair contains flattened image indices and a feature matrix. The
    iterator bounds memory usage during full-resolution patch inference.
    """
    image_array = np.asarray(image)
    _validate_patch_inputs(image_array, patch_size, step)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    height, width = image_array.shape[:2]
    radius = patch_size // 2
    padded = np.pad(
        image_array,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="edge",
    )
    rows, cols = np.mgrid[0:height:step, 0:width:step]
    flat_rows, flat_cols = rows.ravel(), cols.ravel()

    for start in range(0, len(flat_rows), batch_size):
        batch_rows = flat_rows[start : start + batch_size]
        batch_cols = flat_cols[start : start + batch_size]
        patches = np.stack(
            [
                padded[r : r + patch_size, c : c + patch_size]
                for r, c in zip(batch_rows, batch_cols)
            ]
        )
        features = patches.astype(np.float32).reshape(len(patches), -1) / 255.0
        flat_indices = (batch_rows * width + batch_cols).astype(np.int64)
        yield flat_indices, features


def image_to_patches(
    image: ArrayLike,
    labels: ArrayLike,
    target_class_ids: Sequence[int],
    patch_size: int,
    step: int = 1,
    ignore_index: int | None = 0,
) -> tuple[NDArray[np.float32], NDArray[np.uint8], NDArray[np.float32]]:
    """Extract patches centered on valid labeled pixels from one image.

    Returns patch tensors shaped (N, patch_size, patch_size, 3), binary labels,
    and normalized (x, y) center coordinates. Border patches use edge padding.
    """
    image_array = np.asarray(image)
    label_array = np.asarray(labels)
    _validate_patch_inputs(image_array, patch_size, step)
    if label_array.shape != image_array.shape[:2]:
        raise ValueError("labels must match image height and width.")

    height, width = label_array.shape
    rows, cols = np.mgrid[0:height:step, 0:width:step]
    sampled_labels = label_array[::step, ::step]
    valid = np.ones(sampled_labels.shape, dtype=bool)
    if ignore_index is not None:
        valid &= sampled_labels != ignore_index

    selected_rows = rows[valid]
    selected_cols = cols[valid]
    y = binarize_mask(sampled_labels[valid], target_class_ids)

    if selected_rows.size == 0:
        return (
            np.empty((0, patch_size, patch_size, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
            np.empty((0, 2), dtype=np.float32),
        )

    radius = patch_size // 2
    padded = np.pad(
        image_array,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="edge",
    )
    patches = (
        np.stack(
            [
                padded[r : r + patch_size, c : c + patch_size]
                for r, c in zip(selected_rows, selected_cols)
            ]
        ).astype(np.float32)
        / 255.0
    )

    x_pos = selected_cols / max(width - 1, 1)
    y_pos = selected_rows / max(height - 1, 1)
    positions = np.column_stack((x_pos, y_pos)).astype(np.float32)
    return patches, y, positions


def build_patches_dataset(
    dataset,
    indices: Sequence[int],
    target_class_ids: Sequence[int],
    patch_size: int,
    step: int = 1,
    ignore_index: int | None = 0,
) -> tuple[NDArray[np.float32], NDArray[np.uint8], NDArray[np.float32]]:
    """Aggregate patch tensors and labels from selected image indices."""
    if len(indices) == 0:
        raise ValueError("indices cannot be empty.")

    x_parts, y_parts, position_parts = [], [], []
    for index in indices:
        image, labels = dataset.get_pair(int(index))
        X, y, positions = image_to_patches(
            image,
            labels,
            target_class_ids,
            patch_size=patch_size,
            step=step,
            ignore_index=ignore_index,
        )
        if y.size:
            x_parts.append(X)
            y_parts.append(y)
            position_parts.append(positions)

    if not x_parts:
        raise ValueError("No valid labeled pixel centers found in selected images.")

    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(position_parts, axis=0),
    )
