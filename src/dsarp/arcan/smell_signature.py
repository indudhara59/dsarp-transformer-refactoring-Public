"""Stable signatures independent of Arcan vertex identifiers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9.]+", "", str(value or "").casefold().replace("/", "."))


def names(value: object) -> frozenset[str]:
    if isinstance(value, (list, tuple, set, frozenset)): values: Iterable[object] = value
    else: values = re.split(r"[,;|]", str(value or ""))
    return frozenset(filter(None, (normalize_name(item) for item in values)))


@dataclass(frozen=True)
class SmellSignature:
    repository: str
    smell_type: str
    central_component: str
    components: frozenset[str]
    elements: frozenset[str]
    path: str


def signature(record: dict) -> SmellSignature:
    components = names(record.get("affected_component_names", record.get("affected_component_name")))
    return SmellSignature(normalize_name(record.get("repository", record.get("project"))), normalize_name(record.get("smell_type")), normalize_name(record.get("central_component")), components, names(record.get("affected_elements")), normalize_name(record.get("relative_file_path")))
