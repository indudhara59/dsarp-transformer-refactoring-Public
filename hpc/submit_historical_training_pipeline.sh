#!/bin/bash
set -euo pipefail
select_job=$(sbatch --parsable hpc/select_commits.slurm)
mine_job=$(sbatch --parsable --dependency="afterok:${select_job}" hpc/mine_refactorings_array.slurm)
arcan_job=$(sbatch --parsable --dependency="afterok:${mine_job}" hpc/run_arcan_before_after_array.slurm)
dataset_job=$(sbatch --parsable --dependency="afterok:${arcan_job}" hpc/build_historical_dataset.slurm)
transformer_job=$(sbatch --parsable --dependency="afterok:${dataset_job}" hpc/train_integrated_transformer.slurm)
ranker_job=$(sbatch --parsable --dependency="afterok:${transformer_job}" hpc/train_integrated_ranker.slurm)
sbatch --dependency="afterok:${ranker_job}" hpc/evaluate_integrated_models.slurm
