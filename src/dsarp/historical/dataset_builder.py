"""Build candidate-level historical rows and research variants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_variants(frame: pd.DataFrame, output: Path, top_k: int = 3, threshold: float = .35) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        "A": frame,
        "B": frame[frame["overall_improved"] | frame["smell_resolved"]],
        "C": frame[frame["overall_improved"]].sort_values("label_confidence", ascending=False).groupby(["repository", "commit", "smell_id_before"], dropna=False).head(top_k),
        "D": frame[frame["overall_improved"] & frame["is_applicable"]],
        "E": frame[frame["label_confidence"] >= threshold],
        "F": frame[frame["overall_improvement_score"] > 0],
    }
    paths: dict[str, Path] = {}
    manifest: dict[str, Any] = {"recommended": "E", "top_k": top_k, "confidence_threshold": threshold, "variants": {}}
    for name, data in variants.items():
        path = output / f"dataset_{name}.csv"; data.to_csv(path, index=False); paths[name] = path
        manifest["variants"][name] = {"rows": len(data), "repositories": data["repository"].nunique() if len(data) else 0, "commits": data["commit"].nunique() if len(data) else 0, "smells": data["smell_id_before"].nunique() if len(data) else 0, "label_distribution": data["relevance_grade"].value_counts().to_dict() if len(data) else {}}
    (output / "dataset_variant_manifest.json").write_text(json.dumps(manifest, indent=2)); return paths
