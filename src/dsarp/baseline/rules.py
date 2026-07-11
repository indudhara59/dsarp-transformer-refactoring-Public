"""Transparent candidate-specific rule evidence."""
import pandas as pd

RULE_FEATURES = {
    "split_component": ["size_composite", "children_log", "severity_norm", "centrality_composite"],
    "introduce_facade": ["fan_in_norm", "centrality_composite", "page_rank_norm"],
    "dependency_inversion": ["instability_gap_norm", "centrality_composite", "severity_norm"],
    "extract_interface": ["fan_in_norm", "page_rank_norm", "centrality_composite"],
    "extract_mediator": ["fan_in_norm", "fan_out_norm", "hubness_score", "fan_balance"],
    "split_hub_component": ["hubness_score", "size_composite", "severity_norm"],
    "merge_components": ["instability_gap_norm", "size_composite"],
}


def rule_confidence(row: pd.Series) -> tuple[float, dict[str, float]]:
    """Return mean supporting evidence; merge uses strict inverse constraints."""
    features = RULE_FEATURES.get(str(row["recommendation_id"]), ["pressure_score", "severity_norm", "centrality_composite"])
    evidence = {name: float(row.get(name, 0) or 0) for name in features}
    if row["recommendation_id"] == "merge_components":
        evidence = {"low_instability_gap": 1 - evidence["instability_gap_norm"], "small_size": 1 - evidence["size_composite"]}
        score = min(evidence.values()) * 0.6
    else:
        score = sum(evidence.values()) / len(evidence)
    return max(0.0, min(1.0, score)), evidence

