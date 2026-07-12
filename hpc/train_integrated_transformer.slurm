#!/bin/bash
#SBATCH --job-name=dsarp-transformer
#SBATCH --time=REPLACE_TIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --gres=gpu:REPLACE_GPU_COUNT
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
dsarp train-transformer --config configs/training.yaml
