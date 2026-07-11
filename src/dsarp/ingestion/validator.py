"""Input and join-key validation."""
import pandas as pd

from dsarp.exceptions import SchemaError


def duplicate_key_count(frame: pd.DataFrame, keys: list[str]) -> int:
    """Count rows participating as subsequent duplicate identities."""
    return int(frame.duplicated(keys).sum())


def validate_required_values(frame: pd.DataFrame, columns: list[str], role: str) -> None:
    """Reject absent identity values."""
    bad = {column: int(frame[column].isna().sum()) for column in columns if frame[column].isna().any()}
    if bad:
        raise SchemaError(f"Null identity values in {role}: {bad}")

