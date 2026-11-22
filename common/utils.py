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
        

def image_to_pixels(img, labels, target_class_ids, with_position=False):
    """
    image_to_pixels given an image and the wanted true class return the dataset of pixels labeled.

    INPUT
    img: the image to convert
    labels: labels of the image
    target_class_ids: the indexes of the target class returned as label 1
    with_position: if True also position in the features

    RETURNS
    X: the converted image
    y: the labels of the pixels
    """    
    flat_labels = labels.reshape(-1)

    valid = flat_labels != 0

    X = image_features(img, with_position)[valid]

    y = binary_target_mask(flat_labels[valid], target_class_ids)

    return X,y


def build_pixel_dataset(nyu_ds, indeces, target_class_ids,step=1, with_position=False):
    """
    build_pixel_dataset build a dataset of pixels given a dataset of images

    INPUT
    nyu_ds: the dataset of images
    indeces: which images to include in the selection
    target_class_ids: the true class desired
    step: reduced the dimension of the dataset using a simple step slice, used to lower computation.
    with_position: if True also position in the features

    RETURN
    X: the new dataset
    y: the labels
    """
    X_parts, y_parts = [], []

    for i in indeces:
        image, labels = nyu_ds.get_pair(i)
        image = image[::step, ::step]
        labels = labels[::step, ::step]
        
        X,y = image_to_pixels(image,labels,target_class_ids,with_position)
        
        X_parts.append(X)
        y_parts.append(y)

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

def image_to_patches(img, labels, target_class_ids, step, patch_size):
    """
    image_to_patches given an image and the wanted true class return the dataset of pixels labeled with the desired patch around every center pixel.

    INPUT
    img: the image to convert
    labels: labels of the image
    target_class_ids: the indexes of the target class returned as label 1
    step: how many pixel to skip for each center to be chosen
    patch_size: the dimension of the desired patch

    RETURNS
    X: the converted image
    y: the labels of the center pixels
    """    
    #padding
    p = patch_size//2
    padded_img = np.pad(img,((p, p), (p, p), (0, 0)),mode="edge")

    #centers grid
    centers = []
    for r in range(0,img.shape[0],step):
        for c in range(0,img.shape[1],step):
            if labels[r,c] != 0:
                centers.append((r,c))

    X_parts,y_parts = [],[]
    for center in centers:
        X_parts.append(extract_patch(padded_img,center[0]+p,center[1]+p,patch_size))
        y_parts.append(labels[center[0],center[1]])
        
    X = np.stack(X_parts).astype(np.float32) / 255.0
    y = binary_target_mask(y_parts,target_class_ids)
    
    return X,y

def build_patches_dataset(nyu_ds, indeces, target_class_ids, patch_size, step=1):
    """
    build_patches_dataset build a dataset of pixels with a patch given a dataset of images

    INPUT
    nyu_ds: the dataset of images
    indeces: which images to include in the selection
    target_class_ids: the true class desired
    patch_size: dimension of patches
    step: used to subsample the centers of to reduce computations
    
    RETURN
    X: the new dataset
    y: the labels
    """
    X_parts, y_parts = [], []

    for i in indeces:
        image, labels = nyu_ds.get_pair(i)
        
        X,y = image_to_patches(image,labels,target_class_ids,step, patch_size)
        
        X_parts.append(X)
        y_parts.append(y)

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    return X, y