"""Versioned historical contracts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

HISTORICAL_SCHEMA_VERSION = "1.0"


class HistoricalRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    repository: str; commit: str; parent_commit: str; smell_type: str
    smell_id_before: str; smell_id_after: str | None = None
    affected_component: str; candidate_id: str; recommendation_id: str
    smell_transition: str; overall_improvement_score: float; overall_improved: bool
    label_source: Literal["historical_weak", "historical_hybrid"] = "historical_hybrid"
    label_confidence: float; relevance_grade: int; is_applicable: bool
    schema_version: str = HISTORICAL_SCHEMA_VERSION
    score_breakdown: dict[str, float] = Field(default_factory=dict)


INFERENCE_FORBIDDEN_PATTERNS = ("_after", "_delta", "overall_improvement", "smell_resolved", "future_")


def validate_inference_features(features: list[str]) -> None:
    invalid = [name for name in features if any(token in name.casefold() for token in INFERENCE_FORBIDDEN_PATTERNS)]
    if invalid: raise ValueError(f"Target-leakage fields are forbidden in inference features: {invalid}")
