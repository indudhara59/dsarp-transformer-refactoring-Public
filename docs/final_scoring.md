# Final scoring and diversity

The default pre-normalization score is `0.45 ranker + 0.20 transformer + 0.15 LLM + 0.10 rules + 0.10 benefit - 0.15 risk`. Components and penalties are retained in JSONL. Ranker-only, ranker+transformer, full, learned calibration, and tuning modes are configurable; tuning is HPC-only.

Confidence combines transformer calibration, ranker margin, LLM confidence, rule agreement, evidence completeness, and reviewer agreement. Bands are High ≥0.80, Medium ≥0.60, otherwise Low. Exact duplicates, smell/component saturation, family suppression, affected-element overlap, optional embedding cosine similarity, and MMR produce a second diversified list without altering the raw list.

