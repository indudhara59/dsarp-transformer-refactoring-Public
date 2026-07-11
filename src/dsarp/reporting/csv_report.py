"""CSV report writer."""
from pathlib import Path

import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic UTF-8 CSV, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")

