#!/usr/bin/env bash
#SBATCH --job-name=dsarp-train
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --gres=gpu:REPLACE_GPU_TYPE:1
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=REPLACE_EMAIL
set -euo pipefail
module purge
# module load REPLACE_CUDA_MODULE REPLACE_PYTHON_MODULE
: "${DSARP_DATA_DIR:?Set DSARP_DATA_DIR}" "${DSARP_OUTPUT_DIR:?Set DSARP_OUTPUT_DIR}" "${DSARP_MODEL_CACHE:?Set DSARP_MODEL_CACHE}"
export HF_HOME="$DSARP_MODEL_CACHE" TRANSFORMERS_CACHE="$DSARP_MODEL_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}"
srun torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-1}" scripts/train_transformer.py --dataset "$DSARP_OUTPUT_DIR/datasets/training_examples.parquet"
