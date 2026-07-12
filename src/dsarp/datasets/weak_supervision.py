"""Configurable, explicitly non-ground-truth weak labels."""
from dataclasses import asdict, dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class LabelVote:
    label: int | None
    confidence: float
    abstain: bool
    explanation: str


def _threshold(field: str, threshold: float, explanation: str) -> Callable[[pd.Series], LabelVote]:
    def vote(row: pd.Series) -> LabelVote:
        value = row.get(field)
        if pd.isna(value): return LabelVote(None, 0, True, f"{field} missing")
        distance = abs(float(value) - threshold)
        return LabelVote(int(float(value) >= threshold), min(1, .55 + distance), False, explanation)
    return vote


LABEL_FUNCTIONS = {
    "rule_confidence": _threshold("rule_confidence", .6, "Stage 1 rule confidence supports applicability"),
    "applicability": _threshold("applicability_score", .6, "Stage 1 applicability exceeds threshold"),
    "pressure": _threshold("pressure_score", .6, "Smell-specific pressure supports intervention"),
    "benefit": _threshold("expected_benefit", .6, "Expected architectural benefit exceeds threshold"),
    "risk_guard": lambda row: LabelVote(0, float(row.get("estimated_risk", 0)), False, "High predicted change risk") if float(row.get("estimated_risk", 0) or 0) >= .75 else LabelVote(None, 0, True, "Risk guard abstained"),
}


def aggregate_weak_labels(frame: pd.DataFrame, minimum_confidence: float = .55) -> pd.DataFrame:
    """Aggregate label functions by confidence-weighted vote."""
    records = []
    for _, row in frame.iterrows():
        votes = {name: function(row) for name, function in LABEL_FUNCTIONS.items()}
        active = [vote for vote in votes.values() if not vote.abstain and vote.confidence >= minimum_confidence]
        positive = sum(v.confidence for v in active if v.label == 1); total = sum(v.confidence for v in active)
        probability = positive / total if total else float("nan")
        records.append({"candidate_id": row.candidate_id, "is_applicable": int(probability >= .5) if total else float("nan"), "label_confidence": max(probability, 1-probability) if total else 0, "label_source": "weak_supervision", "weak_label_votes": {k: asdict(v) for k, v in votes.items()}})
    return pd.DataFrame(records)

