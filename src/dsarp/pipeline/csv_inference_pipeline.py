"""Primary three-CSV inference entry point (shared by manual and repository modes)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from dsarp.constants import INPUT_FILENAMES
from dsarp.pipeline.full_pipeline import run_full_pipeline


def run_csv_inference(smells: Path, affects: Path, metrics: Path, output_dir: Path, top_k: int = 20, transformer_model: Path | None = None, ranker_model: Path | None = None, baseline_only: bool = False, dry_run: bool = False, force: bool = False) -> dict[str, Path]:
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for role, source in {"smells": smells, "affects": affects, "metrics": metrics}.items():
        target = input_dir / INPUT_FILENAMES[role]
        if source.resolve() != target.resolve(): shutil.copy2(source, target)
    overrides: dict[str, Any] = {"data_dir": str(input_dir), "output_dir": str(output_dir), "top_k": top_k, "dry_run": dry_run, "force": force}
    if not baseline_only: overrides.update({"transformer_model": str(transformer_model) if transformer_model else None, "ranker_model": str(ranker_model) if ranker_model else None})
    paths = run_full_pipeline(overrides=overrides)
    notice = output_dir / "reports" / "inference_context.txt"; notice.parent.mkdir(parents=True, exist_ok=True)
    notice.write_text("Recommendation generated from architecture-level Arcan metrics without source-code context.\n")
    paths["inference_context"] = notice
    return paths
