"""Shared data loading, splitting, and pixel-sampling utilities for NYU Depth V2."""

from __future__ import annotations

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, jaccard_score,classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


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

def sample_pixels(labels, n_pixels, rng):
    """
    sample random pixels
    
    INPUT
    labels: the labels of the pixels
    n_pixels: the number of pixels
    rng: random
    
    OUTPUT
    rows_selected
    cols_selected
    """
    rows, cols = np.nonzero(labels)
    n = min(n_pixels, len(rows))
    chosen = rng.choice(len(rows), size=n, replace=False)
    return rows[chosen], cols[chosen]

def evaluate(y_true, y_proba, threshold=0.5):
    """
    This function is used to evaluate the different models in the same way
    
    INPUT
    y_true: the true labels
    y_proba: the probability of being floor
    threshold: from which probability a pixel is considered floor
    """
    y_pred = (y_proba >= threshold).astype(np.uint8)
    names = ["not floor", "floor"]

    # numbers
    print(f"AUC:      {roc_auc_score(y_true, y_proba):.3f}")
    print(f"AP:       {average_precision_score(y_true, y_proba):.3f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}  (baseline 'always not floor': {1 - y_true.mean():.3f})")
    print(f"IoU:      {jaccard_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred, target_names=names))

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true", display_labels=names, ax=axes[0])
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[1])
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=axes[2])
    axes[0].set_title("Confusion matrix")
    axes[1].set_title("ROC curve")
    axes[2].set_title("Precision-recall curve")
    fig.tight_layout()
    plt.show()

def image_features(img,with_position=False):
    """
    return per-pixel features, both RGB and optionally position in range [0,1]

    INPUT
    img: the image
    with_position: if True return also position in the features
    
    RETURNS
    feats: the image converted in pixels with the features requested
    """
    h, w = img.shape[:2]
    feats = img.reshape(-1, 3).astype(np.float32) / 255.0

    if with_position:
        xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        feats = np.column_stack([feats, xx.ravel(), yy.ravel()]).astype(np.float32)

    return feats


def build_pixel_dataset(nyu_ds, indeces, target_class_ids,n_pixels, seed, with_position=False):
    """
    build_pixel_dataset build a dataset of pixels given a dataset of images

    INPUT
    nyu_ds: the dataset of images
    indeces: which images to include in the selection
    target_class_ids: the true class desired
    n_pixel: number of extracted pixels
    seed: random seed
    with_position: if True also position in the features

    RETURN
    X: the new dataset
    y: the labels
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []

    for i in indeces:
        image, labels = nyu_ds.get_pair(i)
        rows, cols = sample_pixels(labels, n_pixels, rng)

        feats = image_features(image, with_position).reshape(*labels.shape, -1)
        X_parts.append(feats[rows, cols])
        y_parts.append(binary_target_mask(labels[rows, cols], target_class_ids))

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def extract_patch(padded_image, r, c, patch_size):
    """
    Extract a patch of determined size from the given image starting from a given pixel
    
    INPUT
    padded_image: The original image with padding, padding needed to extract the correct patch in pixels that are on the border
    r: row of the selected pixel in the padded image
    c: column of the selected pixel in the padded image
    patch_size: The wanted dimension of the patch, patch_size x patch_size, must be odd
    
    OUTPUT
    patch: cutted image
    """
    if patch_size % 2 == 0: 
        raise ValueError("No even numbers accepted in patch size")
    return padded_image[r-patch_size//2:r+patch_size//2+1,c-patch_size//2:c+patch_size//2+1]

def image_to_patches(img, labels, target_class_ids, n_pixels, patch_size, rng, include_positions=False):
    """
    image_to_patches given an image and the wanted true class return the dataset of pixels labeled with the desired patch around every center pixel.

    INPUT
    img: the image to convert
    labels: labels of the image
    target_class_ids: the indexes of the target class returned as label 1
    n_pixels: how many pixel to extract
    rng: random
    patch_size: the dimension of the desired patch
    include_position: the function can include the position of the centers

    RETURNS
    X: the converted image
    y: the labels of the center pixels
    pos: the positions of the centers if include_position is enabled
    """    
    #padding
    p = patch_size//2
    padded_img = np.pad(img,((p, p), (p, p), (0, 0)),mode="edge")

    #random centers
    rows, cols = sample_pixels(labels, n_pixels, rng)
    centers = list(zip(rows, cols))

    h, w = img.shape[:2]
    X_parts,y_parts, pos_parts = [],[],[]
    
    for center in centers:
        X_parts.append(extract_patch(padded_img,center[0]+p,center[1]+p,patch_size))
        y_parts.append(labels[center[0],center[1]])
        if include_positions:
            pos_parts.append((center[0] / (h - 1),center[1] / (w - 1)))
        
    X = np.stack(X_parts).astype(np.float32) / 255.0
    y = binary_target_mask(y_parts,target_class_ids)
    pos = np.array(pos_parts, dtype=np.float32) if include_positions else None
    return X,y,pos

def build_patches_dataset(nyu_ds, indeces, target_class_ids, patch_size, n_pixels, seed, include_positions=False):
    """
    build_patches_dataset build a dataset of pixels with a patch given a dataset of images

    INPUT
    nyu_ds: the dataset of images
    indeces: which images to include in the selection
    target_class_ids: the true class desired
    patch_size: dimension of patches
    n_pixels: how many pixel to extract
    seed: random seed
    include_position: the function can include the position of the centers
        
    RETURN
    X: the new dataset
    y: the labels
    pos: the position of centers if include_position is enabled
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts, pos_parts = [], [], []

    for i in indeces:
        image, labels = nyu_ds.get_pair(i)
        
        X_curr,y_curr,pos_curr = image_to_patches(image,labels,target_class_ids,n_pixels, patch_size, rng, include_positions)
        
        X_parts.append(X_curr)
        y_parts.append(y_curr)
        pos_parts.append(pos_curr)

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    pos = np.concatenate(pos_parts) if include_positions else None
    return X, y, pos