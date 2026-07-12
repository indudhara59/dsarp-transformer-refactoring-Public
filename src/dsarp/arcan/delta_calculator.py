"""Metric deltas with explicit direction metadata."""
from __future__ import annotations

import math
from typing import Any

LOWER_BETTER = {"severity", "strength", "smell_size", "number_of_edges", "atdi", "atdi_weighted", "smell_extent", "instability_gap", "lines_of_code", "number_of_children"}
CONTEXT = {"fan_in", "fan_out", "instability_metric", "page_rank", "abstractness_metric"}


def calculate_deltas(before: dict[str, Any], after: dict[str, Any] | None) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in sorted(LOWER_BETTER | CONTEXT):
        old, new = before.get(metric), None if after is None else after.get(metric)
        output[f"{metric}_before"], output[f"{metric}_after"] = old, new
        try: output[f"{metric}_delta"] = float(old) - float(new) if new is not None else None
        except (TypeError, ValueError): output[f"{metric}_delta"] = None
        output[f"{metric}_direction"] = "lower_is_better" if metric in LOWER_BETTER else "context_dependent"
    output["resolved"] = after is None
    return output
