"""Infer (never claim detection of) architectural strategies from concrete operations."""
from __future__ import annotations

from typing import Any


def infer_strategies(smell_type: str, refactoring_ids: set[str], deltas: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    inferred = []
    for rule in config.get("mappings", {}).get(smell_type, []):
        required = set(rule.get("any_refactorings", []))
        if required and not required.intersection(refactoring_ids): continue
        metrics = rule.get("positive_deltas", [])
        support = sum(float(deltas.get(f"{metric}_delta") or 0) > 0 for metric in metrics)
        metric_ratio = support / len(metrics) if metrics else 1.0
        confidence = float(rule.get("base_confidence", .5)) * (.5 + .5 * metric_ratio)
        for strategy in rule["strategies"]:
            inferred.append({"recommendation_id": strategy, "mapping_rule": rule["id"], "mapping_confidence": confidence, "detected_low_level_refactorings": sorted(refactoring_ids), "supporting_metric_deltas": {m: deltas.get(f"{m}_delta") for m in metrics}, "strategy_source": "historically_inferred"})
    return inferred
