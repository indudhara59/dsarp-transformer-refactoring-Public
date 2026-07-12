#!/usr/bin/env bash
#SBATCH --job-name=dsarp-llm
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --gres=gpu:REPLACE_GPU_TYPE:REPLACE_GPU_COUNT
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
set -euo pipefail
export HF_HOME="${DSARP_MODEL_CACHE:?}" TRANSFORMERS_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun python scripts/validate_with_llm.py
