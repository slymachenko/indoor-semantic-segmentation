import argparse
import json
import logging
from pathlib import Path

from semseg.data.dataset import NYUDepthV2Dataset
from semseg.data import (
    create_splits,
    save_splits,
    get_splits_filename,
)
from semseg.paths import DATA_RAW_DIR, DATA_SPLITS_DIR, ensure_dirs
from semseg.utils import setup_logger
from semseg.config import load_dataset_config

logger = logging.getLogger(__name__)


def split(
    config_path: Path,
    dataset_dir: Path = DATA_RAW_DIR,
    output_dir: Path = DATA_SPLITS_DIR,
) -> None:
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = load_dataset_config(config_path)

    logger.info(f"Loading dataset from {config_path.absolute()}...")
    dataset = NYUDepthV2Dataset(dataset_dir / ds_cfg.filename)

    try:
        train_indices, val_indices, test_indices = create_splits(
            scene_names=dataset.scene_names,
            validation_fraction=ds_cfg.split.val_fraction,
            test_fraction=ds_cfg.split.test_fraction,
            random_state=ds_cfg.split.random_state,
        )
    finally:
        dataset.close()

    splits_name = get_splits_filename(
        filename=ds_cfg.filename,
        val_fraction=ds_cfg.split.val_fraction,
        test_fraction=ds_cfg.split.test_fraction,
    )

    save_splits(
        DATA_SPLITS_DIR / splits_name,
        train_indices,
        val_indices,
        test_indices,
        random_state=ds_cfg.split.random_state,
    )

    for split_name, indicies in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]:
        logger.info(f"Saved {len(indicies)} {split_name} indices")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset Splitting Pipeline CLI")
    parser.add_argument(
        "--config-path",
        "--config",
        type=Path,
        required=True,
        help="Path to the dataset config.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help="Optional path to the dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_SPLITS_DIR,
        help="Optional save directory for split indices.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logger("split")
    logger.info("RUNNING split.py...")

    args = parse_args()

    try:
        split(**vars(args))
    except Exception as e:
        logger.critical(f"Unexpected exception: {type(e).__name__}: {e}")
        exit(1)


if __name__ == "__main__":
    main()
