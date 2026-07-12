"""Typed multi-task model outputs."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MultiTaskOutput:
    loss: Any | None
    applicability_logits: Any
    relevance_logits: Any
    predicted_benefit: Any
    predicted_risk: Any
    fused_embedding: Any
    uncertainty: Any | None = None
    task_losses: dict[str, Any] = field(default_factory=dict)

