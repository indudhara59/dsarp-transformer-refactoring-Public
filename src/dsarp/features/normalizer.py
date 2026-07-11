"""Reproducible quantile-clipped min-max feature scaling."""
from typing import Any

import numpy as np
import pandas as pd


def robust_minmax(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> tuple[pd.Series, dict[str, Any]]:
    """Clip finite values to quantiles, scale to [0,1], and impute median."""
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(0.0, index=series.index), {"lower": None, "upper": None, "median": None, "missing": int(numeric.isna().sum())}
    lower, upper, median = float(finite.quantile(lower_q)), float(finite.quantile(upper_q)), float(finite.median())
    filled = numeric.fillna(median).clip(lower, upper)
    scaled = pd.Series(0.5 if upper == lower else (filled - lower) / (upper - lower), index=series.index).clip(0, 1)
    return scaled.astype(float), {"lower": lower, "upper": upper, "median": median, "missing": int(numeric.isna().sum())}

