# Integrated historical + CSV pipeline completion record

## Existing modules inspected and reused

Stage 1–3 completion records and the ingestion, canonical schema, filtering, joining, features, candidate taxonomy/generator, baseline, transformer inference, LambdaMART, LLM, ensemble, diversity, full pipeline, reporting, CLI, configuration, tests, and Slurm definitions were inspected. Historical snapshots deliberately reuse `discover_files`, `load_csv`, `map_frame`, `filter_selected_smells`, `join_arcan`, and canonical normalization. Manual inference terminates in the existing `FullPipeline`.

## Added design and modules

`dsarp.mining` adds safe Git/worktree management, configurable deterministic commit selection, cached/mock RefactoringMiner execution, and normalized parsing. `dsarp.arcan` adds cached/mock execution, output validation, Stage 1-backed snapshots, stable signatures, exact/fuzzy ambiguity-aware matching, and directional deltas. `dsarp.historical` adds versioned rows, overlap association, configured smell-specific improvement, two-layer mappings, confidence bands, train-only frequencies, leakage validation, and variants A–F. Integrated, CSV, and repository pipelines plus CLI commands, scripts, configuration, documentation, tests, and an `afterok` Slurm chain were added.

## Schema and matching

Canonical schema 2.0 extends records with repository/commit traceability, affected elements, all requested smell/component metrics, hashes, Arcan/pipeline versions, and schema version while allowing old extra fields. Historical schema 1.0 is candidate-level. Matching never depends solely on vertex ID: smell type, normalized central component, component/element Jaccard, path similarity, and repository identity determine a configurable score. Ambiguous candidates are retained as ambiguous; unmatched before/after smells become resolved/new transitions.

## Improvement, association, taxonomy, and labels

Delta convention is `before - after`; metric direction remains explicit and context-dependent fan/PageRank metrics are interpreted only through smell-specific YAML weights. Refactoring association records element, component, package, source, target, hierarchical-prefix, Jaccard, and matched-element evidence. RefactoringMiner operations are concrete Layer 1 detections. YAML rules infer Layer 2 architectural strategies with mapping confidence and supporting deltas. Weak/hybrid confidence combines configured evidence and penalties and must not be interpreted as causality or expert truth.

## Dataset variants and leakage prevention

Variants A–F are materialized with E recommended. After values, deltas, resolution, overall improvement, and future fields are rejected by inference feature-manifest validation. Frequency models fit only one training partition. `split_group` binds candidates from the same repository/commit/smell. Temporal evaluations must fit on prior commits only. After metrics create labels but never candidate inputs.

## Inference preservation

`dsarp recommend` accepts the three named CSV paths and calls the existing full pipeline; `--baseline-only` preserves operation without learned artifacts. Reports add the required architecture-only notice. `dsarp analyze-repository` resolves a local version, uses a detached worktree, invokes configured Arcan once, and calls the exact same CSV pipeline.

## Lightweight validation performed

`python3 -m compileall -q src scripts tests/historical` completed successfully. Configuration files were parsed during inspection. Dependency-based pytest/CLI execution was attempted but this local interpreter has no project dependencies (`pytest`, `pandas`, `typer`, and Pydantic were unavailable), so no dependency installation or model download was performed. Synthetic tests were added for matching, resolved/new smells, Jaccard, parser sides, leakage rejection, and ambiguity penalties.

## Intentionally not executed

No real repository history mining, cloning/fetching, RefactoringMiner execution, Arcan execution, transformer/ranker training, model download, LLM inference, GPU work, Slurm job, Optuna study, or large experiment was run. No empirical quality metric is claimed.

## Exact HPC sequence

```bash
conda env create -f hpc/environment.yml
conda activate dsarp-stage2
export DSARP_DATA_DIR=/REPLACE/data
export DSARP_OUTPUT_DIR=/REPLACE/outputs
export DSARP_REPOSITORY_CACHE=/REPLACE/repositories
export DSARP_REFACTORINGMINER_PATH=/REPLACE/RefactoringMiner
export DSARP_ARCAN_PATH=/REPLACE/arcan.jar
export DSARP_JAVA_HOME=/REPLACE/java
export DSARP_MODEL_CACHE=/REPLACE/model-cache
export HF_HOME="$DSARP_MODEL_CACHE" TRANSFORMERS_CACHE="$DSARP_MODEL_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Replace every REPLACE_* directive in hpc/*.slurm and configs/*.yaml.
sbatch hpc/submit_historical_training_pipeline.sh
```

Logical order: select commits; RefactoringMiner array; before/after Arcan array; match/build/variants/splits; DistilBERT and hybrid experiments; export embeddings; historical rank data; LambdaMART; evaluation; optional LLM validation; reports.

## Paths

Prepared snapshots: `data/historical/<repository>/<commit>/{metadata.json,refactoringminer.json,before/,after/}`. Selected commits: `outputs/historical/selected_commits.csv`. Candidate labels: `outputs/historical/datasets/historical_candidates.csv`. Variants: `outputs/historical/datasets/variants/dataset_{A..F}.csv`. Stage manifests/errors: `outputs/historical/{manifests,errors}/`. Manual recommendations: `outputs/current-system/final/`. Repository recommendations: `outputs/project-version/final/`.

## Known limitations and remaining experiments

The generic Arcan array script requires a site-specific wrapper that maps each array index to isolated parent/current worktrees because Arcan CLI distributions differ. Historical report aggregation beyond dataset manifests remains an HPC post-processing step. Repository URLs must first be cloned into the configured cache; repository inference currently accepts local checkouts. Empirical thresholds/weights, multi-project temporal splits, expert adjudication, calibration, ablations A–H, causal validity, and final performance all remain experiments.
