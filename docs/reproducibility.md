# Reproducibility

Persist fixed split/qid and feature manifests, reducer and calibration fits, configuration/input hashes, seeds, Git commit, package/GPU/Slurm metadata, model identifiers, Optuna storage/trials, LLM prompt/response hashes, stage manifests, and complete score breakdowns. Normalizers, reducers, calibrators, and feature selection must fit on training data only. Resume skips completed manifests; `--force` recomputes; dry-run writes no stage completion.

