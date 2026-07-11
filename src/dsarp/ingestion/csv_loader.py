"""Defensive CSV loading."""
from pathlib import Path

import pandas as pd

from dsarp.exceptions import SchemaError


def load_csv(path: Path) -> pd.DataFrame:
    """Read CSV values without losing identifier precision."""
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=True, low_memory=False)
    except Exception as exc:
        raise SchemaError(f"Cannot read {path}: {exc}") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise SchemaError(f"Duplicate columns in {path}: {frame.columns[frame.columns.duplicated()].tolist()}")
    return frame

