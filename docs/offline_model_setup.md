# Offline model setup

On an internet-enabled approved machine, obtain the configured Hugging Face model and tokenizer under their licenses, verify checksums, and transfer the complete snapshot to `DSARP_MODEL_CACHE` or a local path. On HPC set `HF_HOME` and `TRANSFORMERS_CACHE` to that directory plus `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Set `model_name_or_path` and `tokenizer_name_or_path` to the transferred path. Loaders default to `local_files_only=True`; tests never call the network. GraphCodeBERT, CodeBERT, MiniLM, and university-hosted compatible AutoModel encoders can be selected without model-specific code.

