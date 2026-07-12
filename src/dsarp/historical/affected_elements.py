"""Association evidence between a smell area and one concrete refactoring."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from dsarp.arcan.smell_matcher import jaccard
from dsarp.arcan.smell_signature import names, normalize_name


def association(smell: dict[str, Any], refactoring: dict[str, Any]) -> dict[str, float | int]:
    smell_elements = names(smell.get("affected_elements")) | names(smell.get("affected_component_name")) | names(smell.get("central_component"))
    source, target = names(refactoring.get("source_elements")), names(refactoring.get("target_elements"))
    ref_elements = source | target | names(refactoring.get("code_elements"))
    paths = {normalize_name(PurePosixPath(str(p)).parent) for p in refactoring.get("file_paths", [])}
    packages = names(smell.get("affected_packages")) | names(smell.get("central_component"))
    exact = smell_elements & ref_elements
    prefix = sum(any(a.startswith(b) or b.startswith(a) for b in packages) for a in paths)
    return {"element_overlap": jaccard(smell_elements, ref_elements), "source_overlap": jaccard(smell_elements, source), "target_overlap": jaccard(smell_elements, target), "package_overlap": jaccard(packages, paths), "hierarchical_prefix_matches": prefix, "matched_elements": len(exact), "component_overlap": jaccard(names(smell.get("affected_component_name")), ref_elements | paths)}
