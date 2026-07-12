#!/bin/bash
#SBATCH --job-name=dsarp-select
#SBATCH --time=REPLACE_TIME
#SBATCH --mem=REPLACE_MEMORY
set -euo pipefail
dsarp select-commits --repository "$DSARP_REPOSITORY_CACHE/REPLACE_REPOSITORY" --output "$DSARP_OUTPUT_DIR/historical/selected_commits.csv"
