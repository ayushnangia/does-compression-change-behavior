#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=exp25lingua
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-08:00
#SBATCH --output=exp25_llmlingua_%j.out
# Frozen LLMLingua-2 prepare, then frozen Qwen3.8 behavior scoring sequentially
# on one H100. Submit after the final exp24 data rebuild for immutable inputs.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}; [ -f "$ROOT/behavior.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
REAL_HOME=$HOME
export HOME=$SCRATCH/compute_home HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
PORT=$((10000 + ${SLURM_JOB_ID:-25} % 50000))
PREFIX=experiments/results/exp25_llmlingua2_$SLURM_JOB_ID
mkdir -p "$HOME/.cache" "$SLURM_TMPDIR/triton-exp25"

# Stage 1: frozen 681MB LLMLingua-2 model. Process exit frees GPU memory.
module load StdEnv/2023
module load gcc cuda python/3.11 python-build-bundle/2026a \
  scipy-stack/2026a arrow/19.0.1
CUDA_VISIBLE_DEVICES=0 $REAL_HOME/ENV-compress2/bin/python \
  experiments/exp25_llmlingua2.py prepare \
  --inputs experiments/results/exp24_qwen38_data/validation.jsonl \
           experiments/results/exp24_qwen38_data/test.jsonl \
  --out ${PREFIX}_prepared.jsonl --rate 0.25 --per-task 2 --device cuda

# Stage 2: known-working Qwen3.8 executor stack on the now-free GPU.
(
  module purge
  module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0
  exec env CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$SLURM_TMPDIR/triton-exp25 \
    $REAL_HOME/ENV-vllm2/bin/vllm serve Qwen/Qwen3.8-27B \
    --port $PORT --served-model-name qwen38-exp25 --max-model-len 77824 \
    --generation-config vllm --gpu-memory-utilization 0.92 \
    --max-num-seqs 8 --max-num-batched-tokens 1024
) > experiments/vllm_exp25_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { tail -80 experiments/vllm_exp25_$SLURM_JOB_ID.log; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo executor-not-ready; exit 1; }

CUDA_VISIBLE_DEVICES=0 $REAL_HOME/ENV-compress2/bin/python \
  experiments/exp25_llmlingua2.py score \
  --prepared ${PREFIX}_prepared.jsonl --out ${PREFIX}_rows.jsonl \
  --summary ${PREFIX}_summary.json --executor-url http://127.0.0.1:$PORT \
  --executor-name qwen38-exp25 --samples 4 --max-tokens 4096
