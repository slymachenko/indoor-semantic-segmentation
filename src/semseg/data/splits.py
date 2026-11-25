from collections import defaultdict
from pathlib import Path
import numpy as np
import json

from collections.abc import Sequence


def create_splits(
    scene_names: Sequence[str],
    validation_fraction: float,
    test_fraction: float,
    random_state: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split image indices while keeping scenes together."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("Invalid validation fraction.")
    if not 0 <= test_fraction < 1:
        raise ValueError("Invalid test fraction.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must sum to less than 1.")

    scene_to_indices = defaultdict(list)
    for index, scene in enumerate(scene_names):
        scene_to_indices[scene].append(index)
    scenes = list(scene_to_indices.keys())

    if len(scenes) < 3:
        raise ValueError("At least three scenes are required.")

    rng = np.random.default_rng(random_state)
    rng.shuffle(scenes)

    n_scenes = len(scenes)
    n_test = int(test_fraction * n_scenes)
    n_val = int(validation_fraction * n_scenes)

    test_scenes = scenes[:n_test]
    val_scenes = scenes[n_test : n_test + n_val]
    train_scenes = scenes[n_test + n_val :]

    test_indicies = [idx for scene in test_scenes for idx in scene_to_indices[scene]]
    val_indicies = [idx for scene in val_scenes for idx in scene_to_indices[scene]]
    train_indicies = [idx for scene in train_scenes for idx in scene_to_indices[scene]]

    rng.shuffle(test_indicies)
    rng.shuffle(val_indicies)
    rng.shuffle(train_indicies)

    return train_indicies, val_indicies, test_indicies


def get_splits_filename(
    filename: str | Path, val_fraction: float, test_fraction: float
):
    return f"{Path(filename).name}-{val_fraction}-{test_fraction}.json"


def save_splits(
    path: str | Path,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    random_state: int,
) -> Path:
    """Persist split indices as a portable JSON file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "random_state": int(random_state),
        "train": [int(i) for i in train_indices],
        "validation": [int(i) for i in val_indices],
        "test": [int(i) for i in test_indices],
    }

    with destination.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return destination


def load_splits(path: str | Path) -> dict[str, list[int]]:
    """Load persisted indices and perform basic consistency checks."""
    source = Path(path)

    with source.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    return (
        [int(i) for i in raw["train"]],
        [int(i) for i in raw["validation"]],
        [int(i) for i in raw["test"]],
    )
