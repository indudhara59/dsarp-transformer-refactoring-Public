"""Deterministic, label-free architecture context text."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import pandas as pd

from dsarp.semantic.template import SECTION_ORDER


@dataclass(frozen=True)
class ContextConfig:
    missing_token: str = "<MISSING>"
    numeric_precision: int = 4
    max_affected_elements: int = 30
    max_characters: int = 12000


FIELDS = {
    "PROJECT": ["project"], "VERSION": ["version_id"], "ARCHITECTURAL_SMELL": ["smell_type"], "AFFECTED_COMPONENT": ["affected_component_name"], "CENTRAL_COMPONENT": ["central_component"], "AFFECTED_ELEMENTS": ["affected_elements"],
    "SMELL_CHARACTERISTICS": ["severity", "strength", "smell_size", "number_of_edges", "instability_gap", "atdi", "atdi_weighted"],
    "COMPONENT_METRICS": ["fan_in", "fan_out", "instability_metric", "lines_of_code", "number_of_children", "page_rank", "abstractness_metric"],
    "ENGINEERED_INDICATORS": ["hubness_score", "god_component_pressure", "unstable_dependency_pressure", "hub_like_dependency_pressure"],
    "PROPOSED_REFACTORING": ["display_name"], "REFACTORING_DESCRIPTION": ["taxonomy_description"], "INTENDED_EFFECT": ["intended_effect"], "CONTRAINDICATIONS": ["contraindications"],
}


def _format(value: Any, config: ContextConfig, *, affected: bool = False) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)): return config.missing_token
    if isinstance(value, float): return f"{value:.{config.numeric_precision}f}"
    if affected:
        if isinstance(value, str):
            try: parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError): parsed = [item.strip() for item in value.strip("[]").split(",")]
        else: parsed = value
        if isinstance(parsed, (list, tuple)): return ", ".join(str(v) for v in parsed[:config.max_affected_elements])
    return str(value)


def build_context(row: pd.Series | dict[str, Any], config: ContextConfig = ContextConfig()) -> str:
    """Build ordered text, excluding every label/comment field by construction."""
    values = row if isinstance(row, dict) else row.to_dict(); sections = []
    for section in SECTION_ORDER:
        lines = []
        for field in FIELDS[section]:
            rendered = _format(values.get(field), config, affected=section == "AFFECTED_ELEMENTS")
            lines.append(rendered if len(FIELDS[section]) == 1 else f"{field}={rendered}")
        sections.append(f"[{section}]\n" + "\n".join(lines))
    text = "\n\n".join(sections)
    if len(text) <= config.max_characters: return text
    # Preserve identity and proposal sections; truncate only the middle evidence.
    prefix = "\n\n".join(sections[:4]); suffix = "\n\n".join(sections[-4:]); budget = max(0, config.max_characters-len(prefix)-len(suffix)-8)
    return prefix + "\n\n" + "\n\n".join(sections[4:-4])[:budget] + "\n\n" + suffix

