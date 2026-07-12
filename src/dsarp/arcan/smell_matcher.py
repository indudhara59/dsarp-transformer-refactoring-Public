"""Exact/fuzzy one-to-one matching with ambiguity retained explicitly."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from dsarp.arcan.smell_signature import SmellSignature, signature


def jaccard(left: set[str] | frozenset[str], right: set[str] | frozenset[str]) -> float:
    if not left and not right: return 1.0
    return len(left & right) / len(left | right) if left | right else 0.0


@dataclass(frozen=True)
class MatchConfig:
    type_weight: float = .35; central_weight: float = .25; components_weight: float = .2
    elements_weight: float = .1; path_weight: float = .1; threshold: float = .6; ambiguity_margin: float = .03


def score(left: SmellSignature, right: SmellSignature, config: MatchConfig) -> float:
    if left.smell_type != right.smell_type: return 0.0
    central = 1.0 if left.central_component and left.central_component == right.central_component else SequenceMatcher(None, left.central_component, right.central_component).ratio()
    path = SequenceMatcher(None, left.path, right.path).ratio() if left.path and right.path else 0.0
    return config.type_weight + config.central_weight * central + config.components_weight * jaccard(left.components, right.components) + config.elements_weight * jaccard(left.elements, right.elements) + config.path_weight * path


def match_smells(before: list[dict[str, Any]], after: list[dict[str, Any]], config: MatchConfig = MatchConfig()) -> list[dict[str, Any]]:
    unused = set(range(len(after))); output: list[dict[str, Any]] = []
    for old in before:
        ranked = sorted(((score(signature(old), signature(after[i]), config), i) for i in unused), reverse=True)
        viable = [(value, index) for value, index in ranked if value >= config.threshold]
        if not viable:
            output.append({"before": old, "after": None, "match_score": 0.0, "match_method": "unmatched", "ambiguity": False, "competing_matches": [], "transition": "resolved", "unmatched_reason": "below_threshold"}); continue
        best, index = viable[0]; competing = [after[i].get("smell_id") for value, i in viable[1:] if best - value <= config.ambiguity_margin]
        ambiguous = bool(competing)
        if not ambiguous: unused.remove(index)
        output.append({"before": old, "after": None if ambiguous else after[index], "match_score": best, "match_method": "fuzzy", "ambiguity": ambiguous, "competing_matches": competing, "transition": "ambiguous" if ambiguous else "persisted", "unmatched_reason": "ambiguous" if ambiguous else None})
    for index in unused:
        output.append({"before": None, "after": after[index], "match_score": 0.0, "match_method": "unmatched", "ambiguity": False, "competing_matches": [], "transition": "newly_introduced", "unmatched_reason": "no_before_match"})
    return output
