# LLM validation

Only top-N ranker candidates are validated. The LLM does not detect smells or replace ranking. Prompts contain supplied Arcan, rule, transformer, taxonomy, ranker, and optional source evidence and demand strict JSON. Responses are schema checked, recommendation-ID checked, retried, hashed, cached, and audited in success/error JSONL files. Failures remain visible and do not stop other candidates.

Backends are disabled, mock, local Hugging Face, vLLM/OpenAI-compatible, and university OpenAI-compatible. No secrets are stored; endpoint keys come from `DSARP_LLM_API_KEY`. Local loaders are offline-only. A confident grounded invalid judgment penalizes but never deletes a candidate.

