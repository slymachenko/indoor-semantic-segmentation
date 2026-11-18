from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGS_DIR = ROOT_DIR / "figs"

DATA_RAW_DIR = DATA_DIR / "raw"

# Put nyu_depth_v2_labeled.mat in ./data/
MAT_PATH = DATA_RAW_DIR / "nyu_depth_v2_labeled.mat"

# Binary task: selected semantic class (1) vs every other labeled class (0).
# The NYU label value 0 is unlabeled and is excluded from sampling/evaluation.
TARGET_CLASS_NAMES = ["floor", "floor mat", "floor trim"]

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
