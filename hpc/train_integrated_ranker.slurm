#!/bin/bash
#SBATCH --job-name=dsarp-ranker
#SBATCH --time=REPLACE_TIME
#SBATCH --mem=REPLACE_MEMORY
set -euo pipefail
python scripts/build_historical_rank_dataset.py --candidates "$DSARP_OUTPUT_DIR/historical/datasets/variants/dataset_E_train.csv" --predictions "$DSARP_OUTPUT_DIR/transformer_predictions/candidate_predictions.csv"
dsarp train-ranker
