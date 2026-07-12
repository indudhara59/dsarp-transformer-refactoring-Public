#!/usr/bin/env bash
#SBATCH --job-name=dsarp-embed
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --gres=gpu:REPLACE_GPU_TYPE:1
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
set -euo pipefail
export HF_HOME="${DSARP_MODEL_CACHE:?}" TRANSFORMERS_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun python scripts/export_embeddings.py --checkpoint models/transformer/best.pt --dataset "$DSARP_OUTPUT_DIR/datasets/training_examples.parquet"
