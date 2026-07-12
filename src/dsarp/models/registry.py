"""Ablation and model metadata registry."""
ABLATIONS = {"numeric_only":{"semantic":False,"numeric":True,"categorical":False},"semantic_only":{"semantic":True,"numeric":False,"categorical":False},"semantic_numeric":{"semantic":True,"numeric":True,"categorical":False},"semantic_numeric_categorical":{"semantic":True,"numeric":True,"categorical":True},"full":{"semantic":True,"numeric":True,"categorical":True}}


def get_ablation(name: str) -> dict[str, bool]:
    if name not in ABLATIONS: raise KeyError(f"Unknown ablation: {name}")
    return ABLATIONS[name].copy()

