"""Numeric Arcan feature MLP."""
import torch
from torch import nn


class NumericEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = .2) -> None:
        super().__init__(); layers: list[nn.Module] = []; previous = input_dim
        for width in hidden_dims: layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)]); previous = width
        layers.append(nn.Linear(previous, output_dim)); self.network = nn.Sequential(*layers); self.output_dim = output_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

