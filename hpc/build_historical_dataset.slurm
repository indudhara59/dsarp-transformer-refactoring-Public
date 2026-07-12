#!/bin/bash
#SBATCH --job-name=dsarp-history-data
#SBATCH --time=REPLACE_TIME
#SBATCH --mem=REPLACE_MEMORY
set -euo pipefail
dsarp build-historical-dataset --data-dir "$DSARP_DATA_DIR/historical" --output-dir "$DSARP_OUTPUT_DIR/historical" --resume
dsarp build-dataset-variants --dataset "$DSARP_OUTPUT_DIR/historical/datasets/historical_candidates.csv" --output-dir "$DSARP_OUTPUT_DIR/historical/datasets/variants"
