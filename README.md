# DSARP: Transformer-Assisted Ranking of Refactoring Recommendations

The integrated system also learns weak, confidence-weighted labels from historical Git changes using RefactoringMiner plus Arcan before/after snapshots. Historical training and current inference share the existing Arcan parser and canonical join; the original Stage 1–3 workflows remain available. See [integrated architecture](docs/integrated_architecture.md), [historical mining](docs/historical_mining.md), and [HPC execution](docs/hpc_historical_pipeline.md).

DSARP starts **after** Arcan detects architectural smells. Stage 1 validates Arcan exports, joins smell evidence to affected components and their metrics, creates smell-appropriate refactoring candidates, and ranks them with a deterministic, interpretable baseline. It does not redetect smells or inspect source code.

## Scope and data flow

Only `godComponent`, `unstableDep`, and `hubLikeDep` (plus documented aliases) enter recommendation generation. Other types, currently including `cyclicDep`, remain visible in filtering statistics.

```text
Arcan CSVs → validation/filtering → composite joins → canonical rows
           → candidates → robust features → baseline scores → diverse rankings/reports
```

Inputs are discovered recursively below `--data-dir` (including `data/` and `data/raw/`):

- `smells-smell-characteristics.csv`: smell identity and characteristics.
- `affects-smell-affects.csv`: smell-to-component edges.
- `metrics-component-metrics.csv`: component identity and metrics.

The observed schema is recorded in `outputs/processed/arcan_schema_report.json`. The current exports join smells to affects on `(project, versionId, vertexId=fromId)`, then affects to metrics on `(project, versionId, toId=vertexId)`. Composite keys prevent cross-project/version collisions. Smell-to-component is intentionally one-to-many; smell and metric entity keys must be unique. Missing optional metrics become `NaN`, are documented in feature statistics, and receive median/default handling during scaling. Rows are never silently discarded: unmatched counts are reports, while unsafe duplicate entities are errors.

## Installation and CLI

Python 3.11+ is required. No model or GPU dependencies are part of Stage 1.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
dsarp inspect --data-dir data
dsarp prepare --data-dir data --output-dir outputs
dsarp baseline --data-dir data --output-dir outputs --top-k 20
dsarp run-stage1 --data-dir data --output-dir outputs --top-k 20
pytest
ruff check .
mypy src/dsarp
```

## Taxonomy and baseline

`configs/recommendation_taxonomy.yaml` defines every recommendation’s compatible smell types, explanation, intended effect, indicators, contraindications, priors, implementation outline, family, and warning. Candidate IDs are SHA-256 prefixes over project, version, smell, component, and recommendation identity.

`configs/baseline_scoring.yaml` contains normalization quantiles, pressure weights, risk weights, final weights, and diversity limits. Features use 5th/95th-percentile clipping followed by min-max scaling. Constant columns map to 0.5 and wholly absent columns to 0. Pressure scores are weighted sums whose weights total 1.0 and reflect the indicators in the project specification. Candidate rules supply recommendation-specific evidence.

All scores are clamped to `[0,1]`. The final score is:

```text
0.25 smell priority + 0.30 applicability + 0.25 expected benefit
+ 0.20 rule confidence - 0.20 estimated risk
```

Ranks break ties by final priority descending, benefit descending, risk ascending, then candidate ID. Global selection limits repeated smells/components and optionally applies maximal marginal relevance using pressure, centrality, size, risk, and recommendation-family similarity.

## Outputs

- `processed/arcan_schema_report.json`, `unified_smell_components.csv`, `refactoring_candidates.csv`, `baseline_feature_statistics.json`
- `rankings/smell_instance_ranking.csv`, `per_smell_candidate_ranking.csv`, `global_ranked_recommendations.csv`
- `reports/ranked_recommendations.json`, `report.html`, `run_metadata.json`, `data_quality_report.json`

The HTML report presents project/version context, data quality, top smells, evidence, scores, implementation outlines, warnings, and raw/normalized metric highlights.

## Extension points

To add a recommendation, add a complete taxonomy entry and, when useful, candidate-specific evidence in `baseline/rules.py`; tests should assert compatibility and score bounds. To add a smell, add its canonical alias, taxonomy compatibility, pressure weights/features, and fixture tests. This deliberately requires code and configuration changes so scope cannot expand silently.

## Limitations and roadmap

Arcan exports describe components rather than class-level cohesion, transactionality, ownership, or runtime behavior. Recommendations therefore need architect review; priors are documented baselines, not learned probabilities. `prepare` currently materializes the complete deterministic artifact set to keep command behavior reproducible. MMR uses a compact numerical similarity rather than learned embeddings.

Stage 2 can build annotation protocols and transformer-ready datasets; Stage 3 can train/evaluate learned ranking models against this baseline on university HPC. **No transformer training, large-model inference, GPU work, downloads, Slurm jobs, or hyperparameter searches are run in this development environment.**

## Stage 2 machine-learning layer

Stage 2 adds partial expert annotations, weak supervision, leakage-safe splitting, deterministic architecture-context text, a hybrid PyTorch/AutoModel multi-task network, calibration/evaluation, offline inference, embeddings, and Slurm orchestration. Install ML dependencies with `pip install -e '.[dev,ml]'`, then use `dsarp build-dataset`; training and inference commands are intended for an offline HPC environment with weights already transferred. See [the dataset guide](docs/stage2_dataset.md), [annotation guide](docs/annotation_guide.md), [architecture](docs/transformer_architecture.md), [evaluation](docs/model_evaluation.md), and [HPC/offline setup](docs/hpc_training.md).

Stage 2 outputs live in `outputs/datasets`, `outputs/transformer_predictions`, `outputs/embeddings`, and `outputs/evaluations`. The Stage 1 CLI and outputs remain unchanged.

## Final Stage 3 workflow

Stage 3 adds three-level LambdaMART ranking, full ranking metrics/bootstrapping, grounded cached top-N LLM validation, optional repository context, calibrated ensemble confidence, traceable LLM rejection penalties, exact deduplication/MMR diversity, case studies, and final CSV/JSON/JSONL/Markdown/HTML reports. The complete entry point is `dsarp run`; it remains operational from the three Arcan CSVs using transparent Stage 1 fallbacks when learned artifacts are unavailable.

```bash
dsarp build-rank-dataset
dsarp train-ranker
dsarp evaluate-ranker
dsarp run --data-dir data --output-dir outputs --top-k 20 --llm-backend disabled
```

For offline HPC execution and optional vLLM validation, see [end-to-end HPC](docs/hpc_end_to_end.md), [learning to rank](docs/learning_to_rank.md), [LLM validation](docs/llm_validation.md), [final scoring](docs/final_scoring.md), [ablations](docs/ablation_study.md), and [reproducibility](docs/reproducibility.md).
