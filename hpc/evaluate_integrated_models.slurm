#!/bin/bash
#SBATCH --job-name=dsarp-evaluate
#SBATCH --time=REPLACE_TIME
#SBATCH --mem=REPLACE_MEMORY
set -euo pipefail
dsarp evaluate-transformer --predictions "$DSARP_OUTPUT_DIR/transformer_predictions/test.csv"
dsarp evaluate-ranker
