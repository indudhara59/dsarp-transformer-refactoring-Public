"""Metrics by architectural and label-provenance slices."""
from collections.abc import Callable
from typing import Any

import pandas as pd


def evaluate_slices(frame: pd.DataFrame, columns: list[str], evaluator: Callable[[pd.DataFrame], dict[str, Any]]) -> dict[str, Any]:
    return {column: {str(value): evaluator(group) for value, group in frame.groupby(column, dropna=False)} for column in columns if column in frame}

