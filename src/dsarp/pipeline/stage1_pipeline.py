"""End-to-end Stage 1 orchestration."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dsarp.baseline.diversity import diverse_top_k
from dsarp.baseline.ranker import build_rankings, score_candidates
from dsarp.candidates.generator import generate_candidates
from dsarp.candidates.taxonomy import load_taxonomy
from dsarp.exceptions import ConfigurationError, DsarpError
from dsarp.features.base import build_features
from dsarp.ingestion.csv_loader import load_csv
from dsarp.ingestion.file_discovery import discover_files
from dsarp.ingestion.joiner import join_arcan
from dsarp.ingestion.schema_inspector import inspect_frame
from dsarp.ingestion.schema_mapper import map_frame
from dsarp.ingestion.validator import validate_required_values
from dsarp.preprocessing.filtering import filter_selected_smells
from dsarp.preprocessing.normalization import normalize_canonical_types
from dsarp.reporting.csv_report import write_csv
from dsarp.reporting.html_report import write_html
from dsarp.reporting.json_report import write_json


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict):
            raise TypeError("top-level value must be a mapping")
        return value
    except Exception as exc:
        raise ConfigurationError(f"Malformed configuration {path}: {exc}") from exc


def inspect_data(data_dir: Path, output_dir: Path = Path("outputs")) -> dict[str, Any]:
    """Inspect real inputs and emit the required schema report."""
    files = discover_files(data_dir)
    raw = {role: load_csv(path) for role, path in files.items()}
    mapped_smells = map_frame(raw["smells"], "smells")
    selected, filtering = filter_selected_smells(mapped_smells)
    mapped = {role: map_frame(raw[role], role) for role in raw}
    affects_selected = mapped["affects"].merge(selected[["project", "version_id", "smell_id"]], on=["project", "version_id", "smell_id"], how="inner")
    try:
        _, joins = join_arcan(selected, affects_selected, mapped["metrics"])
    except DsarpError as exc:
        joins = {"warnings": [str(exc)]}
    report = {
        "files": {role: inspect_frame(files[role], raw[role]) for role in files},
        "unique_smell_types": sorted(raw["smells"]["smellType"].dropna().astype(str).unique().tolist()),
        "selected_smell_counts": filtering["selected_smell_counts"],
        "excluded_smell_counts": {k: v for k, v in filtering["raw_smell_counts"].items() if k not in {"godComponent", "unstableDep", "hubLikeDep"}},
        "filtering": filtering,
        "joins": joins,
        "warnings": joins.get("warnings", []),
    }
    write_json(report, output_dir / "processed/arcan_schema_report.json")
    return report


def run_stage1(data_dir: Path, output_dir: Path = Path("outputs"), top_k: int = 20, config_dir: Path = Path("configs")) -> dict[str, Path]:
    """Run ingestion through diversified ranking and all reports."""
    files = discover_files(data_dir)
    raw = {role: load_csv(path) for role, path in files.items()}
    mapped = {role: map_frame(raw[role], role) for role in raw}
    for role, columns in {"smells": ["project", "version_id", "smell_id"], "affects": ["project", "version_id", "smell_id", "affected_component_id"], "metrics": ["project", "version_id", "affected_component_id"]}.items():
        validate_required_values(mapped[role], columns, role)
    selected, filtering = filter_selected_smells(mapped["smells"])
    if selected.empty:
        raise DsarpError("No God Component, Unstable Dependency, or Hub-Like Dependency records found")
    selected_keys = selected[["project", "version_id", "smell_id"]]
    affects_selected = mapped["affects"].merge(selected_keys, on=["project", "version_id", "smell_id"], how="inner", validate="many_to_one")
    unified, joins = join_arcan(selected, affects_selected, mapped["metrics"])
    if unified.empty:
        raise DsarpError("Selected smells produced no affected-component rows")
    unified = normalize_canonical_types(unified)
    scoring = _load_config(config_dir / "baseline_scoring.yaml")
    featured, statistics = build_features(unified, scoring)
    candidates = generate_candidates(featured, load_taxonomy(config_dir / "recommendation_taxonomy.yaml"))
    scored = score_candidates(candidates, scoring)
    smell_ranking, per_smell = build_rankings(scored)
    global_ranking = diverse_top_k(per_smell, top_k, scoring["diversity"])
    paths = {"unified": output_dir / "processed/unified_smell_components.csv", "candidates": output_dir / "processed/refactoring_candidates.csv", "statistics": output_dir / "processed/baseline_feature_statistics.json", "schema": output_dir / "processed/arcan_schema_report.json", "smells": output_dir / "rankings/smell_instance_ranking.csv", "per_smell": output_dir / "rankings/per_smell_candidate_ranking.csv", "global": output_dir / "rankings/global_ranked_recommendations.csv", "json": output_dir / "reports/ranked_recommendations.json", "html": output_dir / "reports/report.html", "metadata": output_dir / "reports/run_metadata.json", "quality": output_dir / "reports/data_quality_report.json"}
    write_csv(unified, paths["unified"]); write_csv(scored, paths["candidates"]); write_csv(smell_ranking, paths["smells"]); write_csv(per_smell, paths["per_smell"]); write_csv(global_ranking, paths["global"])
    write_json(statistics, paths["statistics"])
    schema = inspect_data(data_dir, output_dir)
    quality = {"filtering": filtering, "joins": joins, "warnings": schema["warnings"]}; write_json(quality, paths["quality"])
    now = datetime.now(UTC).isoformat(); fingerprint = "|".join(f"{p.name}:{p.stat().st_size}" for p in files.values())
    metadata = {"run_id": hashlib.sha256(fingerprint.encode()).hexdigest()[:16], "created_at": now, "input_files": {k: str(v) for k, v in files.items()}, "row_counts": {"unified": len(unified), "candidates": len(scored), "global": len(global_ranking)}, "top_k": top_k, "configuration": {"config_dir": str(config_dir)}}
    write_json(metadata, paths["metadata"]); write_json({"metadata": metadata, "recommendations": _json_records(global_ranking)}, paths["json"])
    projects = _json_records(unified[["project", "version_id"]].drop_duplicates())
    write_html({"metadata": metadata, "projects": projects, "quality": quality, "smells": _json_records(smell_ranking.head(top_k)), "recommendations": _json_records(global_ranking)}, paths["html"])
    return paths

