import os
from pathlib import Path

# src/semseg/paths.py -> project root
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data")).resolve()
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_SPLITS_DIR = DATA_DIR / "splits"

# General
CONFIGS_DIR = ROOT_DIR / "configs"
LOGS_DIR = ROOT_DIR / "logs"

OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", ROOT_DIR / "outputs")).resolve()
FIGS_DIR = OUTPUTS_DIR / "figures"


def ensure_dirs() -> None:
    for directory in (
        DATA_RAW_DIR,
        DATA_SPLITS_DIR,
        FIGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
