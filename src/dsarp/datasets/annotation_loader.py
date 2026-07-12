"""Strict annotation CSV loading."""
from pathlib import Path

import pandas as pd

from dsarp.datasets.annotation_schema import ANNOTATION_COLUMNS, AnnotationRecord
from dsarp.exceptions import SchemaError


def load_annotations(path: Path) -> pd.DataFrame:
    """Validate every nonblank annotation row."""
    frame = pd.read_csv(path, dtype=str).replace({"": None})
    missing = set(ANNOTATION_COLUMNS) - set(frame.columns)
    if missing: raise SchemaError(f"Annotation file missing columns: {sorted(missing)}")
    records = []
    for row in frame[ANNOTATION_COLUMNS].to_dict("records"):
        records.append(AnnotationRecord.model_validate(row).model_dump())
    return pd.DataFrame(records)

