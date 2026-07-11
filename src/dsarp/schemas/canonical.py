"""Typed canonical data contracts."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CanonicalModel(BaseModel):
    """Base allowing documented source-specific extras."""

    model_config = ConfigDict(extra="allow")


class ArcanSmellRecord(CanonicalModel):
    project: str
    version_id: str
    smell_id: str
    smell_type: str
    raw_smell_type: str


class ArcanAffectsRecord(CanonicalModel):
    project: str
    version_id: str
    smell_id: str
    affected_component_id: str


class ArcanComponentMetricRecord(CanonicalModel):
    project: str
    version_id: str
    affected_component_id: str


class UnifiedSmellComponentRecord(CanonicalModel):
    project: str
    version_id: str
    version_index: int | None = None
    version_date: str | None = None
    smell_id: str
    smell_type: str
    raw_smell_type: str
    affected_component_id: str
    affected_component_name: str | None = None
    central_component: str | None = None


class PipelineMetadata(CanonicalModel):
    run_id: str
    created_at: str
    input_files: dict[str, str]
    row_counts: dict[str, int]
    warnings: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)

