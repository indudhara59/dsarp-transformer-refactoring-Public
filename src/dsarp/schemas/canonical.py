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
    repository: str | None = None
    commit: str | None = None
    parent_commit: str | None = None
    affected_elements: list[str] = Field(default_factory=list)
    severity: float | None = None
    strength: float | None = None
    smell_size: float | None = None
    number_of_edges: float | None = None
    smell_extent: float | None = None
    smell_extent_by_weight: float | None = None
    atdi: float | None = None
    atdi_weighted: float | None = None
    page_rank_weighted: float | None = None
    instability_gap: float | None = None
    afferent_affected_ratio: float | None = None
    efferent_affected_ratio: float | None = None
    affected_classes_ratio: float | None = None
    fan_in: float | None = None
    fan_out: float | None = None
    instability_metric: float | None = None
    lines_of_code: float | None = None
    number_of_children: float | None = None
    page_rank: float | None = None
    abstractness_metric: float | None = None
    component_type: str | None = None
    construct_type: str | None = None
    relative_file_path: str | None = None
    input_file_hashes: dict[str, str] = Field(default_factory=dict)
    arcan_version: str | None = None
    pipeline_version: str | None = None
    canonical_schema_version: str = "2.0"


class PipelineMetadata(CanonicalModel):
    run_id: str
    created_at: str
    input_files: dict[str, str]
    row_counts: dict[str, int]
    warnings: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
