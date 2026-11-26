import argparse
import logging
import yaml
from pathlib import Path
import numpy as np

from semseg.data.dataset import NYUDepthV2Dataset
from semseg.data import get_splits_filename, load_splits
from semseg.paths import ensure_dirs
from semseg.utils import setup_logger, save_checkpoint, save_json
from semseg.config import load_dataset_config, load_experiment_config
from semseg.models import build_model
from semseg.features.pixel import build_pixel_dataset
from semseg.features.patch import build_patches_dataset
from semseg.evaluation.metrics import evaluate_binary_segmentation
from semseg.paths import DATA_RAW_DIR, CONFIGS_DIR, DATA_SPLITS_DIR, OUTPUTS_DIR

from semseg.data.dataset import SegmentationDataset
from semseg.config import ExperimentConfig, TaskConfig, cfg2dict
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _target_ids_for(dataset: SegmentationDataset, config: TaskConfig) -> list[int]:
    return [dataset.get_class_id(name) for name in config.target_classes]


def prepare_features(
    dataset: SegmentationDataset,
    indices: list[int],
    experiment_config: ExperimentConfig,
    ignore_index: int,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    sampling = experiment_config.sampling
    if sampling.strategy == "pixel":
        X, y = build_pixel_dataset(
            dataset,
            indices,
            _target_ids_for(dataset, experiment_config.task),
            stride=sampling.stride,
            with_position=sampling.with_position,
            ignore_index=ignore_index,
        )
        return X, y
    X, y, _ = build_patches_dataset(
        dataset,
        indices,
        _target_ids_for(dataset, experiment_config.task),
        patch_size=sampling.patch_size,
        step=sampling.step,
        ignore_index=ignore_index,
    )
    return X.reshape(len(X), -1), y


def train(exp_cfg_path: Path, output_dir: Path) -> None:
    ensure_dirs()
    logger.info(f"Running from config: {exp_cfg_path}...")
    exp_cfg = load_experiment_config(exp_cfg_path)
    ds_cfg = load_dataset_config(CONFIGS_DIR / exp_cfg.dataset_config_path)

    dataset = NYUDepthV2Dataset(DATA_RAW_DIR / ds_cfg.filename)

    try:
        splits_name = get_splits_filename(
            filename=ds_cfg.filename,
            val_fraction=ds_cfg.split.val_fraction,
            test_fraction=ds_cfg.split.test_fraction,
        )

        train_idx, val_idx, test_idx = load_splits(DATA_SPLITS_DIR / splits_name)

        X_train, y_train = prepare_features(
            dataset, train_idx, exp_cfg, ds_cfg.ignore_index
        )
        X_val, y_val = prepare_features(dataset, val_idx, exp_cfg, ds_cfg.ignore_index)

        model = build_model(exp_cfg.model.name, exp_cfg.model.params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val, threshold=exp_cfg.task.threshold)
        val_metrics = evaluate_binary_segmentation(y_val, y_pred, y_proba)

        run_dir = output_dir / exp_cfg.name
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "results").mkdir(exist_ok=True)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg2dict(exp_cfg), sort_keys=False), encoding="utf-8"
        )
        (run_dir / "dataset_config.yaml").write_text(
            yaml.safe_dump(cfg2dict(ds_cfg), sort_keys=False),
            encoding="utf-8",
        )
        save_checkpoint(model, run_dir / "checkpoints" / "model.joblib")
        save_json(val_metrics, run_dir / "results" / "validation_metrics.json")
        save_json(
            {"train": train_idx, "validation": val_idx, "test": test_idx},
            run_dir / "results" / "splits.json",
        )
        logger.info(val_metrics)
    finally:
        dataset.close()

    logger.info(f"Run completed: {exp_cfg.name}")
    logger.info(f"Results saved to {run_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Training Pipeline CLI")
    parser.add_argument(
        "--exp_cfg_path",
        "--config",
        type=Path,
        required=True,
        help="Configuration YAML path or name.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Optional path to outputs directory.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logger("train")
    logger.info("RUNNING train.py...")

    args = parse_args()

    try:
        train(**vars(args))
    except Exception as e:
        # logger.critical(f"Unexpected exception: {type(e).__name__}: {e}")
        logger.exception(f"Unexpected exception: {type(e).__name__}: {e}")
        exit(1)


if __name__ == "__main__":
    main()
