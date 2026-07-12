#!/usr/bin/env bash
#SBATCH --job-name=dsarp-calibrate
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun python scripts/calibrate_transformer.py --validation "$DSARP_OUTPUT_DIR/transformer_predictions/validation_predictions.csv"
