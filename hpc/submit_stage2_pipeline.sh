#!/usr/bin/env bash
# Submit only after replacing every REPLACE_* placeholder and transferring model weights.
set -euo pipefail
dataset=$(sbatch --parsable hpc/build_dataset.slurm)
training=$(sbatch --parsable --dependency="afterok:${dataset}" hpc/train_transformer.slurm)
evaluation=$(sbatch --parsable --dependency="afterok:${training}" hpc/evaluate_transformer.slurm)
calibration=$(sbatch --parsable --dependency="afterok:${evaluation}" hpc/calibrate_transformer.slurm)
embeddings=$(sbatch --parsable --dependency="afterok:${calibration}" hpc/export_embeddings.slurm)
inference=$(sbatch --parsable --dependency="afterok:${embeddings}" hpc/inference_transformer.slurm)
printf 'dataset=%s train=%s eval=%s calibration=%s embeddings=%s inference=%s\n' "$dataset" "$training" "$evaluation" "$calibration" "$embeddings" "$inference"
