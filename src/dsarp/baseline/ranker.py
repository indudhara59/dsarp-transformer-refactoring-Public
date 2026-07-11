"""Baseline scoring and deterministic ranking."""
import json
from typing import Any

import numpy as np
import pandas as pd

from dsarp.baseline.rules import rule_confidence


def score_candidates(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Calculate all six documented [0,1] candidate scores and breakdowns."""
    out = frame.copy()
    rules = out.apply(rule_confidence, axis=1)
    out["rule_confidence"] = rules.map(lambda value: value[0])
    out["rule_evidence"] = rules.map(lambda value: json.dumps(value[1], sort_keys=True))
    out["applicability_score"] = (.45 * out["applicability_prior"] + .55 * out["rule_confidence"]).clip(0, 1)
    out["expected_benefit"] = (.40 * out["benefit_prior"] + .40 * out["pressure_score"] + .20 * out["severity_norm"]).clip(0, 1)
    rw = config["risk_weights"]
    missing_fraction = out[["severity", "strength", "fan_in", "fan_out", "page_rank", "lines_of_code"]].isna().mean(axis=1)
    out["estimated_risk"] = (rw["centrality_composite"] * out.centrality_composite + rw["fan_in_norm"] * out.fan_in_norm + rw["page_rank_norm"] * out.page_rank_norm + rw["size_composite"] * out.size_composite + rw["smell_extent_norm"] * out.smell_extent_norm + rw["taxonomy_prior"] * out.risk_prior + rw["missing_fraction"] * missing_fraction).clip(0, 1)
    out["smell_priority"] = (.55 * out.pressure_score + .25 * out.severity_norm + .20 * out.centrality_composite).clip(0, 1)
    w = config["final_weights"]
    positive = w["smell_priority"] * out.smell_priority + w["applicability"] * out.applicability_score + w["benefit"] * out.expected_benefit + w["rule_confidence"] * out.rule_confidence
    out["final_priority_score"] = (positive - w["risk"] * out.estimated_risk).clip(0, 1)
    out["score_breakdown"] = out.apply(lambda r: json.dumps({"positive": {k: float(r[{"smell_priority": "smell_priority", "applicability": "applicability_score", "benefit": "expected_benefit", "rule_confidence": "rule_confidence"}[k]]) * float(w[k]) for k in ("smell_priority", "applicability", "benefit", "rule_confidence")}, "risk_penalty": float(r.estimated_risk) * float(w["risk"])}, sort_keys=True), axis=1)
    return out


def deterministic_sort(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the specified stable tie-breaking order."""
    return frame.sort_values(["final_priority_score", "expected_benefit", "estimated_risk", "candidate_id"], ascending=[False, False, True, True], kind="stable").reset_index(drop=True)


def build_rankings(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build smell-instance and per-smell candidate rankings."""
    per = deterministic_sort(scored)
    keys = ["project", "version_id", "smell_id"]
    per["candidate_rank_within_smell"] = per.groupby(keys).cumcount() + 1
    smell = per.groupby(keys + ["smell_type"], as_index=False).agg(smell_priority=("smell_priority", "max"), top_candidate_score=("final_priority_score", "max"), affected_components=("affected_component_id", "nunique"))
    smell = smell.sort_values(["smell_priority", "top_candidate_score", "smell_id"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    smell["smell_instance_rank"] = np.arange(1, len(smell) + 1)
    return smell, per

