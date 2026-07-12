"""Atomic model and metadata checkpoints."""
import json
from pathlib import Path
from typing import Any


def save_checkpoint(model: Any, optimizer: Any, path: Path, metadata: dict[str, Any], epoch: int) -> None:
    import torch
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(".tmp")
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"metadata":metadata},temporary); temporary.replace(path)
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True,default=str))


def load_checkpoint(path: Path, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
    import torch
    state=torch.load(path,map_location="cpu",weights_only=False); model.load_state_dict(state["model"])
    if optimizer is not None: optimizer.load_state_dict(state["optimizer"])
    return state

