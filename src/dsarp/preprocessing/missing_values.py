"""Missing-value and numeric parsing policy."""
import re

import numpy as np
import pandas as pd


def parse_numeric(series: pd.Series, *, strict: bool = False) -> pd.Series:
    """Parse common numeric formats, preserving missing values as NaN."""
    cleaned = series.astype("string").str.strip().str.replace(",", "", regex=False)
    cleaned = cleaned.map(lambda x: re.sub(r"%$", "", x) if isinstance(x, str) else x)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if strict:
        invalid = series.notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"Invalid numeric values: {series[invalid].head(5).tolist()}")
    return parsed.astype(float)

