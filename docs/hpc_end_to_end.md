# Offline HPC end-to-end execution

Copy the repository, three Arcan CSVs, annotations, and separately licensed pretrained model snapshots to HPC. Optional repository-mining inputs are `project_metadata.json`, `dependency_edges.csv`, `component_classes.csv`, `change_history.csv`, `source_context.jsonl`, and `historical_refactorings.csv`; the three CSVs remain sufficient.

Create `hpc/environment.yml`, replace every `REPLACE_*`, and export `DSARP_DATA_DIR`, `DSARP_OUTPUT_DIR`, `DSARP_MODEL_CACHE`, `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`. Inspect GPUs with the cluster-approved command, not from login nodes. Submit `hpc/submit_full_pipeline.sh`; use `squeue`, `sacct`, and job logs according to university policy. Stage manifests support resumption; set force only for intentional recomputation. Copy `outputs/final`, `outputs/evaluations`, and model manifests back after completion.

The optional vLLM job accepts model path, tensor parallelism, memory utilization, port, alias, and log variables, checks `/health`, and shuts down via a trap.

