#!/usr/bin/env bash
#SBATCH --job-name=dsarp-vllm
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --gres=gpu:REPLACE_GPU_TYPE:REPLACE_GPU_COUNT
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --mem=REPLACE_MEMORY
#SBATCH --cpus-per-task=REPLACE_CPUS
set -euo pipefail
: "${DSARP_LLM_MODEL:?Set local model path}" "${DSARP_VLLM_PORT:=8000}" "${DSARP_VLLM_ALIAS:=dsarp-validator}" "${DSARP_TENSOR_PARALLEL:=1}" "${DSARP_GPU_MEMORY_UTILIZATION:=0.9}"
log="${DSARP_VLLM_LOG:-hpc/logs/vllm-${SLURM_JOB_ID}.log}"; mkdir -p "$(dirname "$log")"
python -m vllm.entrypoints.openai.api_server --model "$DSARP_LLM_MODEL" --served-model-name "$DSARP_VLLM_ALIAS" --tensor-parallel-size "$DSARP_TENSOR_PARALLEL" --gpu-memory-utilization "$DSARP_GPU_MEMORY_UTILIZATION" --port "$DSARP_VLLM_PORT" >"$log" 2>&1 & server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT TERM INT
for _ in $(seq 1 120); do curl --fail --silent "http://127.0.0.1:${DSARP_VLLM_PORT}/health" && break; sleep 5; done
wait "$server"
