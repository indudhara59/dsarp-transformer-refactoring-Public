#!/usr/bin/env bash
set -euo pipefail
stage1=$(sbatch --parsable hpc/run_stage1.slurm)
transformer=$(sbatch --parsable --dependency="afterok:${stage1}" hpc/inference_transformer.slurm)
embeddings=$(sbatch --parsable --dependency="afterok:${transformer}" hpc/export_embeddings.slurm)
ranker=$(sbatch --parsable --dependency="afterok:${embeddings}" hpc/train_ranker.slurm)
evaluation=$(sbatch --parsable --dependency="afterok:${ranker}" hpc/evaluate_ranker.slurm)
selection=$(sbatch --parsable --dependency="afterok:${evaluation}" hpc/select_top_candidates.slurm)
llm=$(sbatch --parsable --dependency="afterok:${selection}" hpc/run_llm_validation.slurm)
ensemble=$(sbatch --parsable --dependency="afterok:${llm}" hpc/run_final_ensemble.slurm)
diversity=$(sbatch --parsable --dependency="afterok:${ensemble}" hpc/run_diversity.slurm)
report=$(sbatch --parsable --dependency="afterok:${diversity}" hpc/generate_final_report.slurm)
printf 'stage1=%s transformer=%s embeddings=%s ranker=%s eval=%s selection=%s llm=%s ensemble=%s diversity=%s report=%s\n' "$stage1" "$transformer" "$embeddings" "$ranker" "$evaluation" "$selection" "$llm" "$ensemble" "$diversity" "$report"
