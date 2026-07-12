"""Load historical and inference snapshots through the existing Stage 1 parser/joiner."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dsarp.ingestion.csv_loader import load_csv
from dsarp.ingestion.file_discovery import discover_files
from dsarp.ingestion.joiner import join_arcan
from dsarp.ingestion.schema_mapper import map_frame
from dsarp.preprocessing.filtering import filter_selected_smells
from dsarp.preprocessing.normalization import normalize_canonical_types

SCHEMA_VERSION = "2.0"


@dataclass
class ArcanSnapshot:
    records: pd.DataFrame
    input_hashes: dict[str, str]
    schema_version: str = SCHEMA_VERSION


def load_snapshot(directory: Path, repository: str | None = None, commit: str | None = None, parent_commit: str | None = None) -> ArcanSnapshot:
    files = discover_files(directory)
    raw = {role: load_csv(path) for role, path in files.items()}
    mapped = {role: map_frame(frame, role) for role, frame in raw.items()}
    smells, _ = filter_selected_smells(mapped["smells"])
    affects = mapped["affects"].merge(smells[["project", "version_id", "smell_id"]], on=["project", "version_id", "smell_id"], how="inner")
    records, _ = join_arcan(smells, affects, mapped["metrics"])
    records = normalize_canonical_types(records)
    records["repository"] = repository or records["project"]
    records["commit"] = commit
    records["parent_commit"] = parent_commit
    records["canonical_schema_version"] = SCHEMA_VERSION
    hashes = {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in files.items()}
    return ArcanSnapshot(records, hashes)
