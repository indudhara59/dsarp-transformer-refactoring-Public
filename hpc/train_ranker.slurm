#!/usr/bin/env bash
#SBATCH --job-name=dsarp-ranker
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
set -euo pipefail
srun python scripts/build_rank_dataset.py
srun python scripts/train_ranker.py
