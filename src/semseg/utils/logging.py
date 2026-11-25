from pathlib import Path
import logging

from semseg.paths import LOGS_DIR


def setup_logger(
    log_name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(message)s",
    logs_dir: str | Path = LOGS_DIR,
) -> None:
    log_path = Path(logs_dir) / f"{log_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
