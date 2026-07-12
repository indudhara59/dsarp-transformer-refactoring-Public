"""Hybrid semantic, numeric, and categorical multi-task network."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dsarp.models.categorical_encoder import CategoricalEncoder
from dsarp.models.heads import PredictionHeads
from dsarp.models.losses import multitask_loss
from dsarp.models.numeric_encoder import NumericEncoder
from dsarp.models.outputs import MultiTaskOutput
from dsarp.semantic.encoders import masked_mean_pool


class MultimodalRefactoringModel(nn.Module):
    """Fuse optional transformer, numeric, and categorical branches."""
    def __init__(self, transformer: nn.Module | None, semantic_dim: int, numeric_input_dim: int, numeric_hidden_dims: list[int], numeric_output_dim: int, categorical_cardinalities: dict[str, int], categorical_embedding_dim: int, fusion_hidden_dims: list[int], fusion_output_dim: int, dropout: float = .2, relevance_classes: int = 5, relevance_mode: str = "multiclass", uncertainty_head: bool = True, branches: dict[str, bool] | None = None, loss_weights: dict[str, float] | None = None) -> None:
        super().__init__(); self.transformer = transformer; self.branches = branches or {"semantic": transformer is not None, "numeric": True, "categorical": True}; self.relevance_mode = relevance_mode; self.loss_weights = loss_weights or {"applicability":1.,"relevance":1.,"benefit":.5,"risk":.5}
        self.numeric = NumericEncoder(numeric_input_dim, numeric_hidden_dims, numeric_output_dim, dropout); self.categorical = CategoricalEncoder(categorical_cardinalities, categorical_embedding_dim)
        input_dim = (semantic_dim if self.branches["semantic"] else 0) + (numeric_output_dim if self.branches["numeric"] else 0) + (self.categorical.output_dim if self.branches["categorical"] else 0)
        layers: list[nn.Module] = []; previous = input_dim
        for width in fusion_hidden_dims: layers.extend([nn.Linear(previous,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout)]); previous=width
        layers.append(nn.Linear(previous,fusion_output_dim)); self.fusion=nn.Sequential(*layers); self.heads=PredictionHeads(fusion_output_dim,relevance_classes,relevance_mode=="ordinal",uncertainty_head)

    def forward(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, numeric_features: torch.Tensor | None = None, categorical_features: dict[str, torch.Tensor] | None = None, labels: dict[str, torch.Tensor] | None = None) -> MultiTaskOutput:
        pieces=[]
        if self.branches["semantic"]:
            if self.transformer is None or input_ids is None or attention_mask is None: raise ValueError("semantic inputs required")
            encoded=self.transformer(input_ids=input_ids,attention_mask=attention_mask); pieces.append(masked_mean_pool(encoded.last_hidden_state,attention_mask))
        if self.branches["numeric"]:
            if numeric_features is None: raise ValueError("numeric_features required")
            pieces.append(self.numeric(numeric_features))
        if self.branches["categorical"]:
            if categorical_features is None: raise ValueError("categorical_features required")
            pieces.append(self.categorical(categorical_features))
        fused=self.fusion(torch.cat(pieces,dim=-1)); output=MultiTaskOutput(None,self.heads.applicability(fused),self.heads.relevance(fused),self.heads.benefit(fused),self.heads.risk(fused),fused,self.heads.uncertainty(fused) if self.heads.uncertainty else None)
        if labels is not None: output.loss, output.task_losses=multitask_loss(output,labels,self.loss_weights,self.relevance_mode)
        return output

