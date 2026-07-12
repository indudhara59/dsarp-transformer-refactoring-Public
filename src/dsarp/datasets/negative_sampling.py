"""Leakage-safe easy, hard, and counterfactual negatives."""
import hashlib
from typing import Any

import pandas as pd


def sample_negatives(candidates: pd.DataFrame, taxonomy: dict[str, dict[str, Any]], ratios: dict[str, float], seed: int = 42) -> pd.DataFrame:
    """Derive negatives without duplicating a positive candidate identity."""
    positives = set(candidates.candidate_id.astype(str)); records = []
    ordered = candidates.sort_values("candidate_id")
    for kind, ratio in [("easy", ratios.get("easy_ratio", 0)), ("hard", ratios.get("hard_ratio", 0)), ("counterfactual", ratios.get("counterfactual_ratio", 0))]:
        count = round(len(ordered) * float(ratio))
        pool = ordered.sort_values("estimated_risk", ascending=(kind != "easy")).head(max(count, 0))
        for _, row in pool.iterrows():
            record = row.to_dict(); record["negative_type"] = kind; record["is_applicable"] = 0; record["label_source"] = "synthetic_negative"
            if kind == "counterfactual":
                wrong = next((key for key, entry in sorted(taxonomy.items()) if row.smell_type not in entry["smell_types"]), None)
                if wrong is None: continue
                record["recommendation_id"] = wrong
            payload = f"{row.candidate_id}|{kind}|{record['recommendation_id']}|{seed}"
            record["candidate_id"] = "neg_" + hashlib.sha256(payload.encode()).hexdigest()[:20]
            if record["candidate_id"] not in positives: records.append(record)
    return pd.DataFrame(records).drop_duplicates("candidate_id") if records else ordered.iloc[0:0].copy()

