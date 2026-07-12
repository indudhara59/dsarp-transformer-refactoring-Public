"""Ranker feature manifests and train-only embedding reduction."""
from dataclasses import dataclass
from typing import Any

import numpy as np


def configured_features(config:dict[str,Any],include_transformer:bool=True,include_llm:bool=False)->list[str]:
    groups=config["features"]; names=[*groups["raw"],*groups["engineered"],*groups["baseline"],*groups["taxonomy"]]
    if include_transformer: names.extend(groups["transformer"])
    if include_llm: names.extend(["llm_validity","llm_applicability","llm_benefit","llm_risk","semantic_preservation_score","evidence_strength","warning_count"])
    return list(dict.fromkeys(names))


@dataclass
class EmbeddingReducer:
    method:str="pca"; dimensions:int=16; random_state:int=42; reducer:Any=None
    def fit(self,training:np.ndarray)->"EmbeddingReducer":
        if self.method=="none": return self
        if self.method=="pca":
            from sklearn.decomposition import PCA
            self.reducer=PCA(n_components=min(self.dimensions,training.shape[0],training.shape[1]),random_state=self.random_state)
        elif self.method=="svd":
            from sklearn.decomposition import TruncatedSVD
            self.reducer=TruncatedSVD(n_components=min(self.dimensions,training.shape[1]-1),random_state=self.random_state)
        else: raise ValueError(f"Unknown reduction method: {self.method}")
        self.reducer.fit(training); return self
    def transform(self,values:np.ndarray)->np.ndarray: return values if self.method=="none" else self.reducer.transform(values)

