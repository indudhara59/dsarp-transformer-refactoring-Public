"""Expert annotation contracts and aggregation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

ANNOTATION_COLUMNS = ["candidate_id", "is_applicable", "relevance_grade", "expected_benefit", "estimated_risk", "reviewer_id", "annotation_notes"]


class AnnotationRecord(BaseModel):
    """One independent reviewer judgment."""

    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    is_applicable: int | None = None
    relevance_grade: int | None = None
    expected_benefit: float | None = None
    estimated_risk: float | None = None
    reviewer_id: str
    annotation_notes: str = ""

    @field_validator("is_applicable")
    @classmethod
    def binary(cls, value: int | None) -> int | None:
        if value not in (None, 0, 1): raise ValueError("is_applicable must be 0 or 1")
        return value

    @field_validator("relevance_grade")
    @classmethod
    def grade(cls, value: int | None) -> int | None:
        if value is not None and value not in range(5): raise ValueError("relevance_grade must be 0..4")
        return value

    @field_validator("expected_benefit", "estimated_risk")
    @classmethod
    def unit_interval(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1: raise ValueError("score must be in [0,1]")
        return value


def annotation_template(candidate_ids: pd.Series) -> pd.DataFrame:
    """Create a blank, repeatable expert annotation template."""
    return pd.DataFrame({"candidate_id": candidate_ids.drop_duplicates().sort_values(), **{c: "" for c in ANNOTATION_COLUMNS[1:]}})[ANNOTATION_COLUMNS]


def aggregate_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reviewers and quantify disagreement without hiding raw votes."""
    rows: list[dict[str, Any]] = []
    for candidate_id, group in frame.groupby("candidate_id", sort=True):
        applicable = pd.to_numeric(group.is_applicable, errors="coerce").dropna()
        relevance = pd.to_numeric(group.relevance_grade, errors="coerce").dropna()
        benefit = pd.to_numeric(group.expected_benefit, errors="coerce").dropna()
        risk = pd.to_numeric(group.estimated_risk, errors="coerce").dropna()
        vote = float(applicable.mean()) if len(applicable) else np.nan
        agreement = float((applicable.to_numpy()[:, None] == applicable.to_numpy()).mean()) if len(applicable) >= 2 else np.nan
        rows.append({"candidate_id": candidate_id, "is_applicable": int(vote >= .5) if not np.isnan(vote) else np.nan, "relevance_grade": float(relevance.median()) if len(relevance) else np.nan, "expected_benefit_label": float(benefit.mean()) if len(benefit) else np.nan, "estimated_risk_label": float(risk.mean()) if len(risk) else np.nan, "reviewer_count": int(group.reviewer_id.nunique()), "disagreement_score": float(np.mean([applicable.std(ddof=0) if len(applicable) else 0, relevance.std(ddof=0) / 2 if len(relevance) else 0, benefit.std(ddof=0) if len(benefit) else 0, risk.std(ddof=0) if len(risk) else 0])), "applicability_pairwise_agreement": agreement, "label_source": "expert", "label_confidence": float(max(0, 1 - (applicable.std(ddof=0) if len(applicable) else 0)))})
    return pd.DataFrame(rows)
