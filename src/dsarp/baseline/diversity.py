"""Configurable global diversity selection with optional MMR."""
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def diverse_top_k(frame: pd.DataFrame, top_k: int, config: dict[str, Any], *, use_mmr: bool = True) -> pd.DataFrame:
    """Limit dominance and penalize feature/family near-duplicates."""
    if top_k <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    ordered = frame.sort_values(["final_priority_score", "expected_benefit", "estimated_risk", "candidate_id"], ascending=[False, False, True, True], kind="stable").copy()
    max_smell, max_component = int(config["max_per_smell"]), int(config["max_per_component"])
    smell_counts: Counter[str] = Counter(); component_counts: Counter[str] = Counter(); selected: list[int] = []
    features = ["pressure_score", "centrality_composite", "size_composite", "estimated_risk"]
    # Seed distinct smell types when requested and available. Seeding uses the
    # best candidate per type, in score order, and still respects caps.
    minimum = min(int(config.get("min_distinct_smell_types", 0)), ordered.get("smell_type", pd.Series(dtype=str)).nunique(), top_k)
    if minimum:
        seeds = ordered.drop_duplicates("smell_type", keep="first").head(minimum)
        for idx, row in seeds.iterrows():
            if smell_counts[str(row.smell_id)] < max_smell and component_counts[str(row.affected_component_id)] < max_component:
                selected.append(idx)
                smell_counts[str(row.smell_id)] += 1
                component_counts[str(row.affected_component_id)] += 1
    while len(selected) < top_k:
        choices: list[tuple[float, str, int]] = []
        for idx, row in ordered.iterrows():
            if idx in selected or smell_counts[str(row.smell_id)] >= max_smell or component_counts[str(row.affected_component_id)] >= max_component:
                continue
            redundancy = 0.0
            if use_mmr and selected:
                vector = row[features].fillna(0).to_numpy(float)
                similarities = []
                for picked in selected:
                    other = ordered.loc[picked]
                    distance = float(np.linalg.norm(vector - other[features].fillna(0).to_numpy(float))) / 2
                    similarities.append(max(0.0, 1 - distance) * (1.0 if row.recommendation_family == other.recommendation_family else .5))
                redundancy = max(similarities)
            lam = float(config.get("mmr_lambda", .8))
            choices.append((lam * float(row.final_priority_score) - (1 - lam) * redundancy, str(row.candidate_id), idx))
        if not choices:
            break
        winner = sorted(choices, key=lambda x: (-x[0], x[1]))[0][2]
        selected.append(winner); smell_counts[str(ordered.loc[winner, "smell_id"])] += 1; component_counts[str(ordered.loc[winner, "affected_component_id"])] += 1
    result = ordered.loc[selected].reset_index(drop=True)
    result["global_rank"] = np.arange(1, len(result) + 1)
    return result
