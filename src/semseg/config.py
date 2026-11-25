from dataclasses import dataclass, fields, asdict
from pathlib import Path
import yaml

from typing import Any, Optional


class Config:
    def __str__(self) -> str:
        lines = [f"{self.__class__.__name__}({self.name}):"]
        for f in fields(self):
            value = getattr(self, f.name)
            lines.append(f"  {f.name:<16}: {value!r}")
        return "\n".join(lines)


# Dataset Config


@dataclass
class SplitConfig:
    random_state: int
    val_fraction: float
    test_fraction: float


@dataclass
class DatasetConfig(Config):
    name: str
    filename: str
    ignore_index: int | None
    split: SplitConfig


# Experiment Config


@dataclass
class SamplingConfig:
    strategy: str
    stride: int
    with_position: bool


@dataclass
class TaskConfig:
    type: str
    target_classes: list[str]
    class_names: Optional[list[str]]
    threshold: float


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass
class OutputConfig:
    save_checkpoint: bool
    save_predictions: bool


random_state: 42
val_fraction: 0.15
test_fraction: 0.15


@dataclass
class ExperimentConfig(Config):
    name: str
    dataset_config_path: str | Path
    random_state: int
    sampling: SamplingConfig
    task: TaskConfig
    model: ModelConfig
    output: OutputConfig


# Functions


def load_dataset_config(path: Path | str | None = None) -> DatasetConfig:
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    sp = raw["split"]

    return DatasetConfig(
        name=raw["name"],
        filename=raw["filename"],
        ignore_index=raw["ignore_index"],
        split=SplitConfig(**sp),
    )


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        name=raw["name"],
        dataset_config_path=raw["dataset_config_path"],
        random_state=raw["random_state"],
        sampling=SamplingConfig(**raw["sampling"]),
        task=TaskConfig(**raw["task"]),
        model=ModelConfig(**raw["model"]),
        output=OutputConfig(**raw["output"]),
    )


def cfg2dict(config: Config) -> dict[str, Any]:
    return asdict(config)
