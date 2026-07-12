"""Categorical embeddings with explicit unknown indices."""
import torch
from torch import nn


class CategoricalEncoder(nn.Module):
    def __init__(self, cardinalities: dict[str, int], embedding_dim: int) -> None:
        super().__init__(); self.names = list(cardinalities); self.embeddings = nn.ModuleDict({name: nn.Embedding(size + 1, embedding_dim, padding_idx=0) for name, size in cardinalities.items()}); self.output_dim = len(cardinalities) * embedding_dim

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.embeddings[name](values[name]) for name in self.names], dim=-1)

