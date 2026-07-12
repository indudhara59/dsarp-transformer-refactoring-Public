"""Join Stage 1 candidates to semantic inputs and partial labels."""
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dsarp.datasets.annotation_loader import load_annotations
from dsarp.datasets.annotation_schema import aggregate_annotations, annotation_template
from dsarp.datasets.splitters import assert_no_group_leakage, grouped_split
from dsarp.datasets.weak_supervision import aggregate_weak_labels
from dsarp.semantic.context_builder import ContextConfig, build_context


def build_dataset(config_path: Path = Path("configs/dataset.yaml"), transformer_config_path: Path = Path("configs/transformer.yaml")) -> pd.DataFrame:
    """Create one training row per Stage 1 candidate and persist split manifests."""
    config = yaml.safe_load(config_path.read_text()); transformer = yaml.safe_load(transformer_config_path.read_text())
    candidates = pd.read_csv(config["source_candidates"])
    context_config = ContextConfig(**transformer.get("context", {})); candidates["semantic_context_text"] = candidates.apply(lambda row: build_context(row, context_config), axis=1)
    numeric = config["numeric_features"]
    candidates["numeric_feature_vector"] = candidates.apply(lambda row: [float(row.get(c, 0) if pd.notna(row.get(c)) else 0) for c in numeric], axis=1)
    annotations_path = Path(config["annotations"])
    if annotations_path.exists(): labels = aggregate_annotations(load_annotations(annotations_path)); candidates = candidates.merge(labels, on="candidate_id", how="left")
    else:
        template = annotation_template(candidates.candidate_id); path = Path(config["annotation_template"]); path.parent.mkdir(parents=True, exist_ok=True); template.to_csv(path, index=False)
    if config.get("weak_supervision", {}).get("enabled", True):
        weak = aggregate_weak_labels(candidates, config["weak_supervision"].get("minimum_confidence", .55)); candidates = candidates.merge(weak, on="candidate_id", how="left", suffixes=("", "_weak"))
    candidates = grouped_split(candidates, **config["split"]); assert_no_group_leakage(candidates)
    output = Path(config["output"]); output.parent.mkdir(parents=True, exist_ok=True)
    try: candidates.to_parquet(output, index=False)
    except (ImportError, ValueError): candidates.to_csv(output.with_suffix(".csv"), index=False)
    candidates[["candidate_id", "split_group", "split"]].to_csv(output.parent / "split_manifest.csv", index=False)
    return candidates

