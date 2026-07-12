"""Resumable preparation pipeline over already selected repositories and external-tool stages."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from dsarp.arcan.delta_calculator import calculate_deltas
from dsarp.arcan.smell_matcher import MatchConfig, match_smells
from dsarp.arcan.snapshot import load_snapshot
from dsarp.historical.architectural_mapping import infer_strategies
from dsarp.historical.affected_elements import association
from dsarp.historical.improvement import improvement
from dsarp.historical.label_confidence import label_confidence
from dsarp.mining.refactoring_parser import parse_refactoringminer


class HistoricalPipeline:
    def __init__(self, output: Path, configs: Path = Path("configs"), resume: bool = True, dry_run: bool = False, force: bool = False):
        self.output, self.configs, self.resume, self.dry_run, self.force = output, configs, resume, dry_run, force
        self.manifests = output / "manifests"; self.errors = output / "errors"
        self.manifests.mkdir(parents=True, exist_ok=True); self.errors.mkdir(parents=True, exist_ok=True)

    def stage(self, name: str, action: Callable[[], Any]) -> Any:
        marker = self.manifests / f"{name}.json"
        if self.resume and marker.exists() and not self.force and json.loads(marker.read_text()).get("status") == "complete": return None
        if self.dry_run: return None
        try: value = action(); status, error = "complete", None
        except Exception as exc: status, error, value = "failed", str(exc), None; (self.errors / f"{name}.json").write_text(json.dumps({"error": error}, indent=2))
        marker.write_text(json.dumps({"stage": name, "status": status, "finished": datetime.now(UTC).isoformat(), "error": error}, indent=2))
        return value

    def build_commit(self, repository: str, commit: str, parent: str, directory: Path) -> list[dict[str, Any]]:
        before = load_snapshot(directory / "before", repository, parent).records.to_dict("records")
        after = load_snapshot(directory / "after", repository, commit, parent).records.to_dict("records")
        operations = parse_refactoringminer(json.loads((directory / "refactoringminer.json").read_text()))
        mapping = yaml.safe_load((self.configs / "architectural_mapping.yaml").read_text())
        scoring = yaml.safe_load((self.configs / "improvement_scoring.yaml").read_text())
        labels = yaml.safe_load((self.configs / "historical_labels.yaml").read_text())
        result = []
        for transition in match_smells(before, after, MatchConfig(**yaml.safe_load((self.configs / "smell_matching.yaml").read_text())["matching"])):
            old = transition["before"]
            if old is None: continue
            deltas = calculate_deltas(old, transition["after"]); gains = improvement(old["smell_type"], deltas, scoring)
            for operation in operations:
                overlap = association(old, operation)
                strategies = infer_strategies(old["smell_type"], {operation["refactoring_id"]}, deltas, mapping)
                for strategy in strategies:
                    evidence = {**overlap, "smell_prior": labels.get("smell_priors", {}).get(old["smell_type"], .5), "historical_frequency": 0.0, "improvement_score": max(0., gains["overall_improvement_score"]), "mapping_confidence": strategy["mapping_confidence"], "evidence_completeness": gains["evidence_completeness"], "ambiguity_penalty": float(transition["ambiguity"]), "unrelated_refactoring_penalty": float(overlap["element_overlap"] == 0 and overlap["component_overlap"] == 0)}
                    confidence = label_confidence(evidence, labels)
                    result.append({"repository": repository, "commit": commit, "parent_commit": parent, "smell_type": old["smell_type"], "smell_id_before": old["smell_id"], "smell_id_after": transition["after"].get("smell_id") if transition["after"] else None, "affected_component": old.get("affected_component_name"), "candidate_id": f"{repository}:{commit}:{old['smell_id']}:{strategy['recommendation_id']}", "recommendation_id": strategy["recommendation_id"], "smell_transition": transition["transition"], "smell_resolved": deltas["resolved"], **deltas, **gains, **overlap, **strategy, **confidence, "label_source": "historical_hybrid", "schema_version": "1.0", "split_group": f"{repository}:{commit}:{old['smell_id']}"})
        return result

    def run_prepared(self, root: Path) -> Path:
        rows: list[dict[str, Any]] = []
        for metadata in root.glob("*/*/metadata.json"):
            info = json.loads(metadata.read_text()); directory = metadata.parent
            built = self.stage(f"commit-{info['repository']}-{info['commit']}", lambda info=info, directory=directory: self.build_commit(info["repository"], info["commit"], info["parent_commit"], directory))
            if built: rows.extend(built)
        path = self.output / "datasets" / "historical_candidates.csv"; path.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(path, index=False); return path
