#!/usr/bin/env bash
#SBATCH --job-name=dsarp-eval
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --gres=gpu:REPLACE_GPU_TYPE:1
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
set -euo pipefail
# module load REPLACE_MODULES
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="${DSARP_MODEL_CACHE:?}" TRANSFORMERS_CACHE="$HF_HOME"
srun python scripts/evaluate_transformer.py --predictions "$DSARP_OUTPUT_DIR/transformer_predictions/predictions.json"
