"""Offline-safe tokenizer loading."""
from typing import Any


def load_tokenizer(name_or_path: str, *, local_files_only: bool = True, **kwargs: Any) -> Any:
    """Load AutoTokenizer without permitting implicit network access by default."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name_or_path, local_files_only=local_files_only, **kwargs)

