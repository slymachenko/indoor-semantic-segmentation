from pathlib import Path
import joblib
import json

from typing import Any


def save_checkpoint(checkpoint: Any, path: str | Path) -> Path:
    """Save a sklearn/joblib without overwriting a file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()

    if suffix in {".joblib", ".pkl", ".pickle"}:
        with destination.open("xb") as file:
            joblib.dump({"model": checkpoint}, file)
        return destination

    raise ValueError(f"Unsupported checkpoint extension '{suffix}'. Use .joblib, .pkl")


def load_checkpoint(path: str | Path) -> Any:
    """Load a checkpoint."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    suffix = source.suffix.lower()
    if suffix in {".joblib", ".pkl", ".pickle"}:
        payload = joblib.load(source)
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"]
        return payload

    raise ValueError(f"Unsupported checkpoint extension '{suffix}'. Use .joblib, .pkl")


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def save_json(data: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=_json_default)
    return destination


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
