# Stage 2 completion record

## Implemented modules and architecture

Dataset modules cover annotation validation/aggregation, weak votes, negative sampling, deterministic examples, grouped/temporal/leave-one-project-out splits, and persisted manifests. Semantic modules build label-free structured contexts and load configurable AutoTokenizer/AutoModel resources offline. The PyTorch model fuses a transformer mask-mean embedding, numeric MLP, and categorical embeddings, then predicts binary applicability, multiclass/ordinal relevance, sigmoid benefit/risk, uncertainty, and a ranking embedding.

Masked BCE/focal, cross-entropy/ordinal, MSE/Huber losses allow partial labels with default weights 1.0/1.0/0.5/0.5. Calibration supports temperature, Platt, and isotonic methods. Evaluation covers macro/weighted classification, per-class/confusion, ROC/PR AUC, regression correlations/errors, ECE/Brier/reliability, slices, cross-project, and cross-version protocols.

## Label and training-data schema

Required annotation columns are `candidate_id,is_applicable,relevance_grade,expected_benefit,estimated_risk,reviewer_id,annotation_notes`. Applicability is majority vote; relevance median; benefit/risk mean. Disagreement and provenance remain explicit. Every training row retains all required Stage 1 identities, structured text, numeric/categorical inputs, baseline scores, and nullable labels.

## HPC files and execution

Slurm definitions cover dataset building, training, evaluation, calibration, embedding export, and inference. They use explicit resource placeholders, offline Hugging Face variables, scratch/cache variables, and `afterok` submission order. No job, model download, GPU inference, training, or search was executed locally.

```bash
conda env create -f hpc/environment.yml
conda activate dsarp-stage2
export DSARP_DATA_DIR=/REPLACE/data
export DSARP_OUTPUT_DIR=/REPLACE/outputs
export DSARP_MODEL_CACHE=/REPLACE/transferred-model-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Replace all REPLACE_* values in hpc/*.slurm first.
sbatch hpc/submit_stage2_pipeline.sh
```

## Lightweight validation and limitations

Source compilation, configuration structure, shell syntax, deterministic context/split logic, mock-model shapes/loss masking, annotation/weak-label aggregation, and metric/calibration tests are provided. Full dependency-based tests require the declared environment.

Arcan exports lack source bodies and runtime/ownership evidence. GraphCodeBERT therefore encodes architecture-oriented text rather than performing full code understanding. Weak labels are baselines, not truth; expert adjudication and multi-project/version data remain necessary.

## Stage 3

Run controlled HPC experiments, adjudicate annotations, select/calibrate a model on validation data, compare ablations and Stage 1 ranking metrics, integrate learned scores into final ranking, and perform external/project-held-out validation.
