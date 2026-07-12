"""Training example contract."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="allow")
    project: str; version_id: str; smell_id: str; affected_component_id: str
    candidate_id: str; smell_type: str; recommendation_id: str
    semantic_context_text: str
    numeric_features: list[float] = Field(default_factory=list)
    categorical_features: dict[str, str] = Field(default_factory=dict)
    is_applicable: int | None = None
    relevance_grade: int | None = None
    expected_benefit_label: float | None = None
    estimated_risk_label: float | None = None
    expert_priority: float | None = None
    label_source: str | None = None
    label_confidence: float | None = None
    reviewer_id: str | None = None
    annotation_notes: str | None = None

