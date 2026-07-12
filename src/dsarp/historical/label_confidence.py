"""Configurable confidence and conservative quality bands."""
from __future__ import annotations

from typing import Any


def label_confidence(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    weights = config["weights"]
    breakdown = {name: float(weights.get(name, 0)) * float(evidence.get(name, 0)) for name in weights if not name.endswith("penalty")}
    for name in ("ambiguity_penalty", "unrelated_refactoring_penalty"):
        breakdown[name] = -float(weights.get(name, 0)) * float(evidence.get(name, 0))
    value = max(0.0, min(1.0, sum(breakdown.values())))
    thresholds = config["quality_bands"]
    band = "rejected_historical_association"
    for candidate in ("strong_historical_label", "medium_historical_label", "weak_historical_label"):
        if value >= float(thresholds[candidate]): band = candidate; break
    return {"label_confidence": value, "quality_band": band, "score_breakdown": breakdown, "is_applicable": band != "rejected_historical_association", "relevance_grade": 3 if band.startswith("strong") else 2 if band.startswith("medium") else 1 if band.startswith("weak") else 0}
