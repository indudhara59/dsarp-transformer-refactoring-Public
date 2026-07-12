#!/usr/bin/env bash
#SBATCH --job-name=dsarp-stage1
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
set -euo pipefail
srun dsarp run-stage1 --data-dir "$DSARP_DATA_DIR" --output-dir "$DSARP_OUTPUT_DIR"
