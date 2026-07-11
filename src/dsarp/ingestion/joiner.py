"""Cardinality-safe canonical joins."""
from typing import Any

import pandas as pd

from dsarp.exceptions import JoinError
from dsarp.preprocessing.entity_resolver import resolve_component_name

IDENTITY = ["project", "version_id"]


def join_arcan(smells: pd.DataFrame, affects: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one smell-instance × affected-component row without silent loss."""
    smell_keys = [*IDENTITY, "smell_id"]
    metric_keys = [*IDENTITY, "affected_component_id"]
    smell_duplicates = int(smells.duplicated(smell_keys).sum())
    metric_duplicates = int(metrics.duplicated(metric_keys).sum())
    if smell_duplicates or metric_duplicates:
        raise JoinError(f"Non-unique entity keys: smells={smell_duplicates}, metrics={metric_duplicates}")
    affects_entity_keys = [*smell_keys, "affected_component_id"]
    duplicate_affects = int(affects.duplicated(affects_entity_keys).sum())
    # Multiple exported edges to the same component are evidence, not distinct
    # canonical entities. Keep the first stable row and report the collapse.
    affects = affects.drop_duplicates(affects_entity_keys, keep="first")
    smell_affects = smells.merge(affects, on=smell_keys, how="left", validate="one_to_many", indicator="_affects_join", suffixes=("", "_affects"))
    unmatched_smells = int((smell_affects["_affects_join"] == "left_only").sum())
    matched = smell_affects[smell_affects["_affects_join"] == "both"].drop(columns="_affects_join")
    unified = matched.merge(metrics, on=metric_keys, how="left", validate="many_to_one", indicator="_metrics_join", suffixes=("", "_metric"))
    unmatched_components = int((unified["_metrics_join"] == "left_only").sum())
    unified["affected_component_name"] = resolve_component_name(unified)
    unified = unified.drop(columns="_metrics_join").sort_values([*smell_keys, "affected_component_id"], kind="stable").reset_index(drop=True)
    return unified, {
        "unmatched_smells": unmatched_smells,
        "unmatched_affected_components": unmatched_components,
        "duplicate_smell_keys": smell_duplicates,
        "duplicate_metric_keys": metric_duplicates,
        "duplicate_smell_component_edges_collapsed": duplicate_affects,
        "unified_rows": len(unified),
    }
