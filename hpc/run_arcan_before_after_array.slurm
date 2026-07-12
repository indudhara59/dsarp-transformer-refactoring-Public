#!/bin/bash
#SBATCH --job-name=dsarp-arcan
#SBATCH --array=REPLACE_ARRAY_RANGE
#SBATCH --time=REPLACE_TIME
#SBATCH --mem=REPLACE_MEMORY
set -euo pipefail
# Site wrapper should checkout the array row's parent/current commits and invoke configs/arcan_runner.yaml.
python scripts/run_arcan_snapshots.py --config configs/arcan_runner.yaml --dry-run
