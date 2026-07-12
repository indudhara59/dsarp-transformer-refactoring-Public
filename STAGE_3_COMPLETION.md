# Stage 3 completion record

## Final architecture

Stage 3 implements smell, candidate, and global ranking groups; XGBoost LambdaMART `rank:ndcg`; confidence-weighted labels; train-only embedding reduction; fixed/project/temporal split support; all specified ranking metrics and group-bootstrap intervals; required baselines and ablation manifests; built-in XGBoost importance and optional SHAP.

Top-N LLM validation supports disabled, mock, local-HF, vLLM/OpenAI-compatible, and university-compatible endpoints. Prompts are evidence-bounded; JSON is repaired conservatively then Pydantic-validated, retried, cached by complete evidence/decoding hash, and logged without hiding errors. Optional source adapters enrich but never replace the three Arcan inputs.

The default normalized ensemble uses ranker 0.45, transformer 0.20, LLM 0.15, rules 0.10, benefit 0.10, and risk penalty 0.15. Confident grounded invalidity applies a visible penalty and retains rejected rows. Exact deduplication, family/smell/component limits, affected overlap, embedding cosine similarity, and λ=0.75 MMR create a separate diversified ranking.

## Final outputs

`outputs/final/` contains ranked and diversified CSVs, JSON, rejected candidates, smell/per-smell ranks, score JSONL, Markdown explanations, HTML, experiment summary, run metadata, and model manifest. Case studies live under `outputs/evaluations/case_studies`.

## HPC order and commands

The dependency chain is Stage 1 → transformer inference → embedding export → rank dataset/training → evaluation → LLM top candidates → ensemble/diversity → report. All scripts use placeholders and offline caches.

```bash
conda env create -f hpc/environment.yml
conda activate dsarp-stage2
export DSARP_DATA_DIR=/REPLACE/data
export DSARP_OUTPUT_DIR=/REPLACE/outputs
export DSARP_MODEL_CACHE=/REPLACE/model-cache
export HF_HOME="$DSARP_MODEL_CACHE" TRANSFORMERS_CACHE="$DSARP_MODEL_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Replace all REPLACE_* fields first.
sbatch hpc/submit_full_pipeline.sh
```

No transformer/ranker training, Optuna search, large LLM inference, model download, Slurm submission, or long evaluation was run locally. Only compilation, configuration, shell validation, and dependency-available lightweight tests are appropriate here.

## Limitations and project status

The system cannot prove semantic preservation, compilation, tests, runtime improvement, or source-level cohesion without external evidence. LLM judgment is advisory and traceable. Weak labels are not primary truth. Multi-project expert annotations and HPC experiments remain empirical work, not implementation gaps. The three-stage software architecture is complete and Stage 1/2 behavior is preserved.
