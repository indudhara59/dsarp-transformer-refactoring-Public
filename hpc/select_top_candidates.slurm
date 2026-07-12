#!/usr/bin/env bash
#SBATCH --job-name=dsarp-top
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
set -euo pipefail
# Ranker evaluation persists scores; LLM validation selects top-N per project/version.
srun dsarp validate-with-llm
