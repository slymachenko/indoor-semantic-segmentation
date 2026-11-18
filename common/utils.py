"""Shared data loading, splitting, and pixel-sampling utilities for NYU Depth V2."""

from __future__ import annotations

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from collections import defaultdict


class NYUDepthV2:
    def __init__(self, mat_path):
        self.file = h5py.File(mat_path, "r")

        self.images = self.file["images"]
        self.labels = self.file["labels"]

        self.n_images = self.images.shape[0]

        # Decode MATLAB class names.
        self.class_names = []
        for ref in self.file["names"][0]:
            chars = self.file[ref][()].ravel()
            name = "".join(chr(int(c)) for c in chars if c != 0)
            self.class_names.append(name)

    def get_pair(self, index):
        image = self.images[index].transpose(2, 1, 0)
        mask = self.labels[index].T

        return image, mask

    def get_class_id(self, class_name):
        return self.class_names.index(class_name) + 1

    def close(self):
        self.file.close()

    def __len__(self):
        return self.n_images


def binary_target_mask(labels, target_class_ids):
    return np.isin(labels, target_class_ids).astype(np.uint8)


def make_data_splits(
    nyu_dataset: NYUDepthV2,
    validation_fraction: float,
    test_fraction: float,
    random_state: int = 42,
):
    # Retrieve scene names
    scene_refs = nyu_dataset.file["scenes"][0]

    scene_names = []
    for ref in scene_refs:
        char_array = nyu_dataset.file[ref][:]
        scene_name = "".join(chr(c[0]) for c in char_array)
        scene_names.append(scene_name)

    # Group image indices by their unique scene
    scene_to_indices = defaultdict(list)
    for idx, scene in enumerate(scene_names):
        scene_to_indices[scene].append(idx)

    unique_scenes = list(scene_to_indices.keys())

    # Shuffle the unique scenes, not the individual images
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_scenes)

    # Calculate split sizes based on the number of SCENES
    n_scenes = len(unique_scenes)
    n_test = int(test_fraction * n_scenes)
    n_val = int(validation_fraction * n_scenes)

    # Allocate scenes to splits
    test_scenes = unique_scenes[:n_test]
    val_scenes = unique_scenes[n_test : n_test + n_val]
    train_scenes = unique_scenes[n_test + n_val :]

    # Map the split scenes back to their original image indices
    test_indices = [idx for scene in test_scenes for idx in scene_to_indices[scene]]
    val_indices = [idx for scene in val_scenes for idx in scene_to_indices[scene]]
    train_indices = [idx for scene in train_scenes for idx in scene_to_indices[scene]]

    # Shuffle the final indices so your dataloader doesn't process one room sequentially
    rng.shuffle(test_indices)
    rng.shuffle(val_indices)
    rng.shuffle(train_indices)

    return train_indices, val_indices, test_indices
