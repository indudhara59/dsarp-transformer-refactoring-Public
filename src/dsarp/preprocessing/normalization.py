"""Canonical numeric preparation."""
import pandas as pd

from dsarp.preprocessing.missing_values import parse_numeric

NUMERIC_COLUMNS = ["severity", "strength", "atdi", "atdi_weighted", "smell_size", "number_of_edges", "smell_extent", "smell_extent_by_weight", "pagerank_weighted", "instability_gap", "afferent_affected_ratio", "efferent_affected_ratio", "affected_classes_ratio", "fan_in", "fan_out", "instability_metric", "lines_of_code", "number_of_children", "page_rank", "abstractness_metric"]


def normalize_canonical_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse all available canonical numeric fields consistently."""
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column not in result:
            result[column] = float("nan")
        result[column] = parse_numeric(result[column])
    if "version_index" in result:
        result["version_index"] = pd.to_numeric(result["version_index"], errors="coerce").astype("Int64")
    return result

