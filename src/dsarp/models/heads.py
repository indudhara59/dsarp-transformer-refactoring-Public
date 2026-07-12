"""Multi-task prediction heads."""
from torch import nn


class PredictionHeads(nn.Module):
    def __init__(self, input_dim: int, relevance_classes: int = 5, ordinal: bool = False, uncertainty: bool = True) -> None:
        super().__init__(); self.ordinal = ordinal; self.applicability = nn.Linear(input_dim, 1); self.relevance = nn.Linear(input_dim, relevance_classes - 1 if ordinal else relevance_classes); self.benefit = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid()); self.risk = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid()); self.uncertainty = nn.Sequential(nn.Linear(input_dim, 1), nn.Softplus()) if uncertainty else None

