"""Recommendation taxonomy loading and validation."""
from pathlib import Path
from typing import Any

import yaml

from dsarp.exceptions import ConfigurationError

REQUIRED_FIELDS = {"display_name", "smell_types", "description", "intended_effect", "indicators", "contraindications", "applicability_prior", "benefit_prior", "risk_prior", "implementation_outline", "warning", "family"}


def load_taxonomy(path: Path = Path("configs/recommendation_taxonomy.yaml")) -> dict[str, dict[str, Any]]:
    """Load and validate taxonomy entries."""
    try:
        payload = yaml.safe_load(path.read_text())
        entries = payload["recommendations"]
    except Exception as exc:
        raise ConfigurationError(f"Malformed taxonomy {path}: {exc}") from exc
    for key, value in entries.items():
        missing = REQUIRED_FIELDS - set(value)
        if missing:
            raise ConfigurationError(f"Taxonomy entry {key} missing {sorted(missing)}")
        for prior in ("applicability_prior", "benefit_prior", "risk_prior"):
            if not 0 <= float(value[prior]) <= 1:
                raise ConfigurationError(f"{key}.{prior} must be in [0,1]")
    return entries

