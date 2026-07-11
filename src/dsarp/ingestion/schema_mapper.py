"""Map exact Arcan columns to the canonical schema."""
from typing import Mapping

import pandas as pd

from dsarp.exceptions import SchemaError

COMMON = {"project": "project", "versionId": "version_id", "versionIndex": "version_index", "versionDate": "version_date"}
MAPS: dict[str, dict[str, str]] = {
    "smells": {**COMMON, "vertexId": "smell_id", "smellType": "raw_smell_type", "CentralComponent": "central_component", "AffectedElements": "affected_elements", "Severity": "severity", "Strength": "strength", "Size": "smell_size", "NumberOfEdges": "number_of_edges", "SmellExtent": "smell_extent", "SmellExtentByWeight": "smell_extent_by_weight", "ATDI": "atdi", "ATDI_WEIGHTED": "atdi_weighted", "PageRankWeighted": "pagerank_weighted", "InstabilityGap": "instability_gap", "AfferentAffectedRatio": "afferent_affected_ratio", "EfferentAffectedRatio": "efferent_affected_ratio", "AffectedClassesRatio": "affected_classes_ratio"},
    "affects": {**COMMON, "fromId": "smell_id", "toId": "affected_component_id", "to": "affected_component_name"},
    "metrics": {**COMMON, "vertexId": "affected_component_id", "name": "metric_component_name", "FanIn": "fan_in", "FanOut": "fan_out", "InstabilityMetric": "instability_metric", "LinesOfCode": "lines_of_code", "NumberOfChildren": "number_of_children", "PageRank": "page_rank", "AbstractnessMetric": "abstractness_metric", "ComponentType": "component_type", "constructType": "construct_type", "filePathRelative": "relative_file_path"},
}
REQUIRED = {"smells": {"project", "versionId", "vertexId", "smellType"}, "affects": {"project", "versionId", "fromId", "toId"}, "metrics": {"project", "versionId", "vertexId"}}


def map_frame(frame: pd.DataFrame, role: str, overrides: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Rename known columns while requiring identity fields."""
    missing = REQUIRED[role] - set(frame.columns)
    if missing:
        raise SchemaError(f"Unsupported {role} schema; missing columns: {sorted(missing)}")
    mapping = {**MAPS[role], **(overrides or {})}
    return frame.rename(columns=mapping).copy()

