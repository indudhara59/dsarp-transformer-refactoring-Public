"""Feature pipeline."""
from typing import Any

import numpy as np
import pandas as pd

from dsarp.features.normalizer import robust_minmax

NORMALIZE = {"severity": "severity_norm", "strength": "strength_norm", "atdi": "atdi_norm", "atdi_weighted": "atdi_weighted_norm", "smell_size": "smell_size_norm", "number_of_edges": "edge_count_norm", "smell_extent": "smell_extent_norm", "smell_extent_by_weight": "smell_extent_weighted_norm", "pagerank_weighted": "pagerank_weighted_norm", "instability_gap": "instability_gap_norm", "afferent_affected_ratio": "afferent_ratio_norm", "efferent_affected_ratio": "efferent_ratio_norm", "affected_classes_ratio": "affected_classes_ratio_norm", "fan_in": "fan_in_norm", "fan_out": "fan_out_norm", "page_rank": "page_rank_norm", "lines_of_code": "loc_norm", "number_of_children": "children_norm"}


def build_features(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate smell, component, interaction, and pressure features."""
    out = frame.copy()
    stats: dict[str, Any] = {}
    q = config.get("normalization", {})
    for source, target in NORMALIZE.items():
        out[target], stats[target] = robust_minmax(out[source], float(q.get("lower_quantile", .05)), float(q.get("upper_quantile", .95)))
    out["loc_log"] = np.log1p(out["lines_of_code"].fillna(0).clip(lower=0)); out["loc_log"], stats["loc_log"] = robust_minmax(out["loc_log"])
    out["children_log"] = np.log1p(out["number_of_children"].fillna(0).clip(lower=0)); out["children_log"], stats["children_log"] = robust_minmax(out["children_log"])
    out["fan_total"] = out["fan_in_norm"] + out["fan_out_norm"]
    out["fan_product"] = out["fan_in_norm"] * out["fan_out_norm"]
    out["fan_balance"] = 1 - (out["fan_in_norm"] - out["fan_out_norm"]).abs() / (out["fan_total"] + 1e-9)
    out["centrality_composite"] = (out["fan_in_norm"] + out["fan_out_norm"] + out["page_rank_norm"]) / 3
    out["size_composite"] = (out["loc_log"] + out["children_log"] + out["smell_size_norm"]) / 3
    out["severity_strength"] = out["severity_norm"] * out["strength_norm"]
    out["severity_atdi"] = out["severity_norm"] * out["atdi_norm"]
    out["fan_in_pagerank"] = out["fan_in_norm"] * out["page_rank_norm"]
    out["fan_out_instability"] = out["fan_out_norm"] * out["instability_metric"].fillna(.5).clip(0, 1)
    out["instability_gap_severity"] = out["instability_gap_norm"] * out["severity_norm"]
    out["children_loc"] = out["children_log"] * out["loc_log"]
    out["hubness_score"] = (out["fan_in_norm"] + out["fan_out_norm"] + out["fan_balance"] + out["edge_count_norm"] + out["page_rank_norm"]) / 5
    for smell, weights in config["pressure_weights"].items():
        name = {"godComponent": "god_component_pressure", "unstableDep": "unstable_dependency_pressure", "hubLikeDep": "hub_like_dependency_pressure"}[smell]
        out[name] = sum(out[column].fillna(0) * float(weight) for column, weight in weights.items()).clip(0, 1)
    out["pressure_score"] = np.select([out.smell_type.eq("godComponent"), out.smell_type.eq("unstableDep")], [out.god_component_pressure, out.unstable_dependency_pressure], default=out.hub_like_dependency_pressure)
    return out, stats

