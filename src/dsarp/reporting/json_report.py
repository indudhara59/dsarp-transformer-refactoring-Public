"""JSON report writer."""
import json
from pathlib import Path
from typing import Any


def write_json(payload: Any, path: Path) -> None:
    """Write standards-compliant, stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False), encoding="utf-8")

