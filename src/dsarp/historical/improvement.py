"""Configuration-driven smell-specific improvement scoring."""
from __future__ import annotations

from typing import Any


def improvement(smell_type: str, deltas: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    settings = config["smells"][smell_type]
    contributions: dict[str, float] = {}
    available = 0.0
    for metric, weight in settings["weights"].items():
        value = deltas.get(f"{metric}_delta")
        if value is None: continue
        available += abs(float(weight)); scale = float(settings.get("scales", {}).get(metric, 1.0)) or 1.0
        contributions[metric] = max(-1.0, min(1.0, float(value) / scale)) * float(weight)
    if deltas.get("resolved"): contributions["resolution"] = float(settings.get("resolution_bonus", 0.0)); available += abs(contributions["resolution"])
    score = sum(contributions.values()) / available if available else 0.0
    completeness = available / (sum(abs(float(x)) for x in settings["weights"].values()) + abs(float(settings.get("resolution_bonus", 0))))
    return {"metric_contributions": contributions, "overall_improvement_score": score, "overall_improved": score >= float(settings.get("improved_threshold", .1)), "improvement_confidence": max(0.0, min(1.0, completeness)), "evidence_completeness": max(0.0, min(1.0, completeness)), "resolved": bool(deltas.get("resolved"))}
