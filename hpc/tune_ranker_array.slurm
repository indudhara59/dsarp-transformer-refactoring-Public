#!/usr/bin/env bash
#SBATCH --job-name=dsarp-tune
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --array=0-REPLACE_ARRAY_MAX
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
set -euo pipefail
srun python scripts/tune_ranker.py --trial-offset "$SLURM_ARRAY_TASK_ID" --trials "${DSARP_TRIALS_PER_TASK:-10}"
