#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=exp24grpo
#SBATCH --gpus-per-node=h100:2
#SBATCH --time=1-00:00
#SBATCH --output=exp24_grpo_%j.out
# Usage after Gate 1: sbatch exp24_job.sh 10   # plumbing pilot only
# Main run:          sbatch exp24_job.sh -1
# Final frozen executor: Qwen3.8-27B bf16. The 4B model only selects indices.
set -euo pipefail

MAX_STEPS=${1:-10}
DATA_DIR=${2:-experiments/results/exp24_qwen38_data}
MIN_TRAIN=1000
[ "$MAX_STEPS" != "-1" ] && MIN_TRAIN=20  # labeled plumbing pilot only
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}
[ -f "$ROOT/experiments/exp24_grpo_train.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
module load gcc cuda python/3.11 arrow/19.0.1 2>/dev/null
REAL_HOME=$HOME
export HOME=$SCRATCH/compute_home HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$ROOT
mkdir -p "$HOME/.cache" "$SLURM_TMPDIR/triton-vllm" "$SLURM_TMPDIR/triton-train"

# Frozen executor on GPU 1. Training never updates or co-locates with it.
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=$SLURM_TMPDIR/triton-vllm \
  $REAL_HOME/ENV-vllm2/bin/vllm serve Qwen/Qwen3.8-27B \
  --port 8001 --served-model-name qwen38-27b-exp24 --max-model-len 32768 \
  --generation-config vllm --gpu-memory-utilization 0.92 --max-num-seqs 16 \
  > experiments/vllm_exp24_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8001/health >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { tail -80 experiments/vllm_exp24_$SLURM_JOB_ID.log; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:8001/health >/dev/null || { echo 'executor failed readiness'; exit 1; }

# Selector LoRA training on GPU 0. Dr-GRPO removes completion-length bias;
# sequence-level importance weights match the sequence-level selector reward.
EVAL_ARGS=()
[ -s "$DATA_DIR/validation.jsonl" ] && EVAL_ARGS=(--eval-file "$DATA_DIR/validation.jsonl")
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$SLURM_TMPDIR/triton-train \
  $REAL_HOME/ENV-compress2/bin/python experiments/exp24_grpo_train.py \
  --train-file "$DATA_DIR/train.jsonl" "${EVAL_ARGS[@]}" \
  --out experiments/results/exp24_grpo_$SLURM_JOB_ID \
  --reward-log experiments/results/exp24_grpo_rewards_$SLURM_JOB_ID.jsonl \
  --max-steps "$MAX_STEPS" --min-train-examples "$MIN_TRAIN"
