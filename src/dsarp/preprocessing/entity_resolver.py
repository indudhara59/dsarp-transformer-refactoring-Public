"""Component entity resolution helpers."""
import pandas as pd


def resolve_component_name(frame: pd.DataFrame) -> pd.Series:
    """Prefer affects-edge names and fall back to metric names."""
    left = frame.get("affected_component_name", pd.Series(index=frame.index, dtype="object"))
    right = frame.get("metric_component_name", pd.Series(index=frame.index, dtype="object"))
    return left.fillna(right)

