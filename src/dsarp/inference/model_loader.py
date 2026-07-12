"""Offline checkpoint loader with metadata validation."""
import json
from pathlib import Path
from typing import Any,Callable


def load_model(checkpoint:Path,factory:Callable[[dict[str,Any]],Any])->tuple[Any,dict[str,Any]]:
    import torch
    metadata_path=checkpoint.with_suffix(".metadata.json"); metadata=json.loads(metadata_path.read_text()); model=factory(metadata); state=torch.load(checkpoint,map_location="cpu",weights_only=False); model.load_state_dict(state.get("model",state)); return model,metadata

