"""Smell alias normalization and scope filtering."""
import re

import pandas as pd

from dsarp.constants import SELECTED_SMELLS

ALIASES = {
    "godcomponent": "godComponent",
    "unstabledep": "unstableDep",
    "unstabledependency": "unstableDep",
    "hublikedep": "hubLikeDep",
    "hublikedependency": "hubLikeDep",
}


def normalize_smell_type(value: object) -> str | None:
    """Normalize case and punctuation variants into the three canonical names."""
    if value is None or pd.isna(value):
        return None
    key = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    return ALIASES.get(key)


def filter_selected_smells(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep in-scope smells and return traceable filtering statistics."""
    result = frame.copy()
    result["smell_type"] = result["raw_smell_type"].map(normalize_smell_type)
    raw_counts = result["raw_smell_type"].fillna("<null>").value_counts().sort_index().to_dict()
    selected = result[result["smell_type"].isin(SELECTED_SMELLS)].copy()
    return selected, {
        "total": len(result),
        "selected": len(selected),
        "excluded": len(result) - len(selected),
        "raw_smell_counts": raw_counts,
        "selected_smell_counts": selected["smell_type"].value_counts().sort_index().to_dict(),
    }

