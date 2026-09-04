#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=exp24grpo
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=1-00:00
#SBATCH --output=exp24_grpo_%j.out
# Usage after Gate 1: sbatch exp24_job.sh 10   # plumbing pilot only
# Main run:          sbatch exp24_job.sh -1 DATA_DIR SEED
# Exact continuation: sbatch exp24_job.sh -1 DATA_DIR SEED auto ORIGINAL_JOB_ID
# Trillium allocates either 1 or a multiple of 4 GPUs. We request one node;
# GPU0 trains, GPU1 serves the frozen executor, GPUs2-3 remain unused.
# Final frozen executor: Qwen3.8-27B bf16. The 4B model only selects indices.
set -euo pipefail

MAX_STEPS=${1:-10}
DATA_DIR=${2:-experiments/results/exp24_qwen38_data}
SEED=${3:-42}
RESUME=${4:-}
# Continuations must write into the original run directory/log. Fresh jobs
# default to their own Slurm ID; resume jobs pass the original run ID.
RUN_ID=${5:-$SLURM_JOB_ID}
MIN_TRAIN=1000
[ "$MAX_STEPS" != "-1" ] && MIN_TRAIN=20  # labeled plumbing pilot only
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}
[ -f "$ROOT/experiments/exp24_grpo_train.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
# StdEnv changes MODULEPATH; on a pristine --export=NONE shell it must be a
# separate transaction before hierarchical Python modules are resolved.
module load StdEnv/2023
module load gcc cuda python/3.11 python-build-bundle/2026a \
  scipy-stack/2026a arrow/19.0.1
REAL_HOME=$HOME
export HOME=$SCRATCH/compute_home HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
# Preserve Alliance's sitecustomize path: it expands EBPYTHONPREFIXES for
# module-provided typing_extensions/numpy/etc. Never replace PYTHONPATH.
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
mkdir -p "$HOME/.cache" "$SLURM_TMPDIR/triton-vllm" "$SLURM_TMPDIR/triton-train"

# Fail before the 27B startup if the selector's clean-node dependency stack is
# incomplete. Pilot 813697 otherwise spent ten minutes loading vLLM first.
CUDA_VISIBLE_DEVICES=0 $REAL_HOME/ENV-compress2/bin/python - <<'PY'
import typing_extensions, numpy, packaging, torch, transformers, peft, trl, datasets
print("selector dependency preflight green")
PY

# Frozen executor on GPU 1. Training never updates or co-locates with it.
# ENV-vllm2 is Python 3.12 and intentionally relies on Alliance's opencv
# module for packaging/psutil/etc. Keep that module stack in an isolated
# subshell: the selector ENV-compress2 below requires the Python 3.11 stack.
(
  module purge
  module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0 2>/dev/null
  exec env CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=$SLURM_TMPDIR/triton-vllm \
    $REAL_HOME/ENV-vllm2/bin/vllm serve Qwen/Qwen3.8-27B \
    --port 8001 --served-model-name qwen38-27b-exp24 --max-model-len 32768 \
    --generation-config vllm --gpu-memory-utilization 0.92 --max-num-seqs 16
) > experiments/vllm_exp24_$SLURM_JOB_ID.log 2>&1 &
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
OUT=experiments/results/exp24_grpo_$RUN_ID
REWARD_LOG=experiments/results/exp24_grpo_rewards_$RUN_ID.jsonl
RESUME_ARGS=()
if [ "$RESUME" = auto ]; then
  LATEST=$(find "$OUT" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sort -V | tail -1)
  [ -n "$LATEST" ] || { echo "resume requested but no checkpoint in $OUT"; exit 1; }
  RESUME_ARGS=(--resume-from-checkpoint "$LATEST")
  echo "resuming original run $RUN_ID from $LATEST"
elif [ -n "$RESUME" ]; then
  [ -d "$RESUME" ] || { echo "resume checkpoint missing: $RESUME"; exit 1; }
  RESUME_ARGS=(--resume-from-checkpoint "$RESUME")
fi
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$SLURM_TMPDIR/triton-train \
  $REAL_HOME/ENV-compress2/bin/python experiments/exp24_grpo_train.py \
  --train-file "$DATA_DIR/train.jsonl" "${EVAL_ARGS[@]}" \
  --out "$OUT" --reward-log "$REWARD_LOG" \
  --max-steps "$MAX_STEPS" --min-train-examples "$MIN_TRAIN" \
  --seed "$SEED" "${RESUME_ARGS[@]}"
