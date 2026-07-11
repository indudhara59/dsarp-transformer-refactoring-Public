"""Schema and quality inspection."""
from pathlib import Path
from typing import Any

import pandas as pd


def inspect_frame(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize an input file without mutating it."""
    candidate_keys = [c for c in ["project", "versionId", "vertexId", "fromId", "toId", "edgeId"] if c in frame]
    duplicates = {key: int(frame.duplicated([key]).sum()) for key in candidate_keys}
    composite = [c for c in ["project", "versionId", "vertexId"] if c in frame]
    return {
        "file_name": path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": frame.columns.tolist(),
        "inferred_data_types": {c: str(t) for c, t in frame.dtypes.items()},
        "null_counts": {c: int(v) for c, v in frame.isna().sum().items()},
        "duplicate_counts": {"full_rows": int(frame.duplicated().sum()), "single_keys": duplicates, "candidate_composite_key": int(frame.duplicated(composite).sum()) if composite else None},
        "candidate_join_keys": candidate_keys,
        "warnings": [],
    }
