"""Deterministic smell-compatible candidate generation."""
import hashlib
from typing import Any

import pandas as pd


def stable_candidate_id(row: pd.Series, recommendation_id: str) -> str:
    """Hash stable identifying fields into a reproducible identifier."""
    fields = [row.get(k, "") for k in ("project", "version_id", "smell_id", "affected_component_id")]
    payload = "\x1f".join([*(str(v) for v in fields), recommendation_id])
    return "cand_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def generate_candidates(unified: pd.DataFrame, taxonomy: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Generate every compatible recommendation for every canonical row."""
    records: list[dict[str, Any]] = []
    for _, row in unified.sort_values(["project", "version_id", "smell_id", "affected_component_id"], kind="stable").iterrows():
        for recommendation_id, entry in sorted(taxonomy.items()):
            if row["smell_type"] not in entry["smell_types"]:
                continue
            record = row.to_dict()
            record.update({"candidate_id": stable_candidate_id(row, recommendation_id), "recommendation_id": recommendation_id, "display_name": entry["display_name"], "taxonomy_description": entry["description"], "intended_effect": entry["intended_effect"], "implementation_outline": entry["implementation_outline"], "warning_text": entry["warning"], "recommendation_family": entry["family"], "applicability_prior": float(entry["applicability_prior"]), "benefit_prior": float(entry["benefit_prior"]), "risk_prior": float(entry["risk_prior"])})
            records.append(record)
    return pd.DataFrame.from_records(records)

