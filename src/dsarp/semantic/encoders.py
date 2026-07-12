"""Configurable Hugging Face encoder wrapper."""
from typing import Any


def load_auto_model(name_or_path: str, *, local_files_only: bool = True, **kwargs: Any) -> Any:
    """Load AutoModel; callers must transfer weights to the offline cache first."""
    from transformers import AutoModel
    return AutoModel.from_pretrained(name_or_path, local_files_only=local_files_only, **kwargs)


def masked_mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean-pool non-padding token vectors."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)

