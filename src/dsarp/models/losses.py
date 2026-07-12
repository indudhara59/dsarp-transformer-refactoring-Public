"""Masked multi-task losses for partial annotations."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional


def masked_mean(losses: torch.Tensor, targets: torch.Tensor) -> torch.Tensor | None:
    mask = torch.isfinite(targets)
    return losses[mask].mean() if mask.any() else None


def focal_binary(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2) -> torch.Tensor:
    base = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none"); probability = torch.sigmoid(logits); pt = probability * targets + (1-probability)*(1-targets)
    return ((1-pt).pow(gamma) * base)


def ordinal_targets(grades: torch.Tensor, classes: int = 5) -> torch.Tensor:
    thresholds = torch.arange(classes-1, device=grades.device).unsqueeze(0)
    return (grades.unsqueeze(1) > thresholds).float()


def multitask_loss(outputs: Any, labels: dict[str, torch.Tensor], weights: dict[str, float], relevance_mode: str = "multiclass", classification_loss: str = "cross_entropy", regression_loss: str = "huber", class_weights: torch.Tensor | None = None, focal_gamma: float = 2) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Compute only tasks with at least one finite label."""
    losses: dict[str, torch.Tensor] = {}
    app = labels.get("is_applicable")
    if app is not None:
        mask = torch.isfinite(app)
        if mask.any():
            raw = focal_binary(outputs.applicability_logits.squeeze(-1)[mask], app[mask], focal_gamma) if classification_loss == "focal" else functional.binary_cross_entropy_with_logits(outputs.applicability_logits.squeeze(-1)[mask], app[mask], reduction="none")
            losses["applicability"] = raw.mean()
    rel = labels.get("relevance_grade")
    if rel is not None:
        mask = torch.isfinite(rel)
        if mask.any():
            losses["relevance"] = functional.binary_cross_entropy_with_logits(outputs.relevance_logits[mask], ordinal_targets(rel[mask], outputs.relevance_logits.shape[1]+1)) if relevance_mode == "ordinal" else functional.cross_entropy(outputs.relevance_logits[mask], rel[mask].long(), weight=class_weights)
    for task, output_name, label_name in [("benefit", "predicted_benefit", "expected_benefit_label"), ("risk", "predicted_risk", "estimated_risk_label")]:
        target = labels.get(label_name)
        if target is not None:
            mask = torch.isfinite(target)
            if mask.any():
                prediction = getattr(outputs, output_name).squeeze(-1)[mask]; losses[task] = functional.huber_loss(prediction, target[mask]) if regression_loss == "huber" else functional.mse_loss(prediction, target[mask])
    total = sum(weights[name] * value for name, value in losses.items()) if losses else None
    return total, losses

