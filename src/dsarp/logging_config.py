"""Logging setup."""
import logging.config
from pathlib import Path

import yaml


def configure_logging(path: Path | None = None) -> None:
    """Configure structured, non-debug-print logging."""
    config_path = path or Path("configs/logging.yaml")
    if config_path.exists():
        logging.config.dictConfig(yaml.safe_load(config_path.read_text()))
    else:
        logging.basicConfig(level=logging.INFO)
