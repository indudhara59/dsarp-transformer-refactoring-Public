"""Recommendation contracts."""
from typing import Any

from pydantic import BaseModel, Field


class RefactoringCandidate(BaseModel):
    candidate_id: str
    project: str
    version_id: str
    smell_id: str
    smell_type: str
    affected_component_id: str
    recommendation_id: str
    display_name: str
    taxonomy: dict[str, Any] = Field(default_factory=dict)
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    normalized_metrics: dict[str, float] = Field(default_factory=dict)
    feature_values: dict[str, float] = Field(default_factory=dict)
    scoring_breakdown: dict[str, Any] = Field(default_factory=dict)


class RankedRecommendation(RefactoringCandidate):
    rule_confidence: float
    applicability_score: float
    expected_benefit: float
    estimated_risk: float
    smell_priority: float
    final_priority_score: float
    global_rank: int | None = None

