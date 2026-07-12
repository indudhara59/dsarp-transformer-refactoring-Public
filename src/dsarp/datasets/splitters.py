"""Leakage-safe grouped and cross-domain splitters."""
from __future__ import annotations

import hashlib
from typing import Iterator

import pandas as pd


def _group_key(frame: pd.DataFrame, strategy: str) -> pd.Series:
    mapping = {"project": ["project"], "project_version": ["project", "version_id"], "smell": ["project", "version_id", "smell_id"], "project_version_smell": ["project", "version_id", "smell_id"]}
    if strategy not in mapping: raise ValueError(f"Unknown split strategy: {strategy}")
    return frame[mapping[strategy]].astype(str).agg("\x1f".join, axis=1)


def grouped_split(frame: pd.DataFrame, strategy: str = "project_version_smell", train: float = .7, validation: float = .15, test: float = .15, seed: int = 42) -> pd.DataFrame:
    """Assign whole identity groups deterministically to train/validation/test."""
    if not abs(train + validation + test - 1) < 1e-9: raise ValueError("split fractions must total 1")
    result = frame.copy(); keys = _group_key(result, strategy)
    def assign(key: str) -> str:
        value = int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:12], 16) / float(16**12)
        return "train" if value < train else "validation" if value < train + validation else "test"
    result["split"] = keys.map(assign); result["split_group"] = keys
    return result


def assert_no_group_leakage(frame: pd.DataFrame) -> None:
    """Fail when one persisted split group occurs in multiple partitions."""
    leaked = frame.groupby("split_group").split.nunique()
    if (leaked > 1).any(): raise ValueError(f"Split leakage in {int((leaked > 1).sum())} groups")


def leave_one_project_out(frame: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield deterministic leave-one-project-out folds."""
    for project in sorted(frame.project.dropna().unique()): yield str(project), frame[frame.project != project].copy(), frame[frame.project == project].copy()


def temporal_split(frame: pd.DataFrame, test_fraction: float = .2) -> pd.DataFrame:
    """Hold out the latest versions within each project."""
    result = frame.copy(); result["split"] = "train"
    for _, group in result.groupby("project"):
        versions = group[["version_id", "version_index", "version_date"]].drop_duplicates().sort_values(["version_index", "version_date"], na_position="first")
        held = set(versions.tail(max(1, round(len(versions)*test_fraction))).version_id)
        result.loc[group.index[group.version_id.isin(held)], "split"] = "test"
    return result
