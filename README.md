# Indoor Semantic Segmentation

This project explores semantic segmentation on the [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html). We establish the simplest baselines and build on top of them in order to analyze the development of semantic segmentation methods.

## Setup

### Dataset

Download the labeled NYU Depth V2 file from the dataset page and place it on the following path:

```text
data/raw/nyu_depth_v2_labeled.mat
```

The loader expects the labeled MATLAB file with its images, labels, class names, and scene names. Dataset files are not tracked by Git. By default, data is read from `data/`. However, it is possible to set `DATA_DIR` before running the scripts to use a different data root. The expected subdirectory is `raw/`.

### Project

Python 3.10 or newer is required. From the repository root, install the package and its dependencies:

```bash
python -m pip install -e .
```

For notebook support, install the optional development dependencies:

```bash
python -m pip install -e ".[dev]"
```

> [!NOTE]
> The repository also includes a Conda environment specification in [`environment.yaml`](environment.yaml):
>
> ```bash
> conda env create -f environment.yml
> conda activate semseg
> ```

## Run the baselines

Generate the train, validation, and test splits:

```bash
python scripts/split.py --config configs/datasets/nyu_depth_v2.yaml
```

Splits are grouped by scene to prevent images from the same scene leaking between partitions. The default configuration uses a random seed of `42`, with 15% of scenes assigned to validation and 15% to test. The generated JSON is saved under `data/splits/`.
Split generation does not overwrite an existing JSON file at that path; preserve or move an existing split before regenerating it.

Train the desired configuration:

```bash
# RGB features
python scripts/train.py --config configs/experiments/pixel_logreg_rgb.yaml

# RGB features plus pixel position
python scripts/train.py --config configs/experiments/pixel_logreg_rgbxy.yaml
```

## Configuration

- [`configs/datasets/nyu_depth_v2.yaml`](configs/datasets/nyu_depth_v2.yaml) specifies the dataset filename, ignored label, random seed, and split fractions.
- [`configs/experiments/pixel_logreg_rgb.yaml`](configs/experiments/pixel_logreg_rgb.yaml) samples every sixth pixel and uses normalized RGB values.
- [`configs/experiments/pixel_logreg_rgbxy.yaml`](configs/experiments/pixel_logreg_rgbxy.yaml) adds normalized image coordinates to the RGB features.
- [`configs/experiments/patch_logreg_15x15_raw.yaml`](configs/experiments/patch_logreg_15x15_raw.yaml) describes an experimental 15×15 patch setup.

## Outputs

By default, the project writes:

- `outputs/<experiment-name>/checkpoints/model.joblib`: trained model
- `outputs/<experiment-name>/results/validation_metrics.json`: validation metrics
- `outputs/<experiment-name>/results/splits.json`: indices used for train, validation, and test
- `outputs/<experiment-name>/*.yaml`: copies of the configs used for the run
- `logs/split.log` and `logs/train.log`: command logs

Set `OUTPUTS_DIR` before training to write run artifacts elsewhere. Data, logs, and outputs are excluded from Git. \
The validation report includes ROC AUC, average precision, accuracy, intersection over union (IoU), and a confusion matrix.

## Project layout

- `src/semseg/data/dataset`: NYU Depth V2 loading
- `src/semseg/data/`: label conversion, and scene-level splitting
- `src/semseg/features/`: pixel and patch feature preparation
- `src/semseg/models/`: segmentation model interface and model factory
- `src/semseg/evaluation/`: binary segmentation metrics
- `scripts/`: split and training command-line entry points
- `notebooks/`: exploratory analysis and baseline experiments
- `configs/`: dataset and experiment YAML files
