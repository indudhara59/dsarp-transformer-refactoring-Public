"""Normalize RefactoringMiner JSON without assigning commit-level operations to every smell."""
from __future__ import annotations

import re
from typing import Any


def _id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def parse_refactoringminer(payload: dict[str, Any]) -> list[dict[str, Any]]:
    commits = payload.get("commits", [payload])
    parsed: list[dict[str, Any]] = []
    for commit in commits:
        for item in commit.get("refactorings", []):
            left, right = item.get("leftSideLocations", []), item.get("rightSideLocations", [])
            locations = left + right
            parsed.append({
                "refactoring_id": _id(str(item.get("type", "unknown"))),
                "refactoring_type": item.get("type", "Unknown"),
                "description": item.get("description", ""),
                "left_locations": left, "right_locations": right,
                "file_paths": sorted({str(x.get("filePath", "")) for x in locations if x.get("filePath")}),
                "code_elements": sorted({str(x.get("codeElement", "")) for x in locations if x.get("codeElement")}),
                "source_elements": [x.get("codeElement", "") for x in left],
                "target_elements": [x.get("codeElement", "") for x in right],
            })
    return parsed
