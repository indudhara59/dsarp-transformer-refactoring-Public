#!/bin/bash
#SBATCH --job-name=dsarp-rm
#SBATCH --array=REPLACE_ARRAY_RANGE
#SBATCH --time=REPLACE_TIME
#SBATCH --mem=REPLACE_MEMORY
set -euo pipefail
python scripts/mine_refactorings.py --commits "$DSARP_OUTPUT_DIR/historical/selected_commits.csv" --repository "$DSARP_REPOSITORY_CACHE/REPLACE_REPOSITORY" --output-dir "$DSARP_DATA_DIR/historical"
