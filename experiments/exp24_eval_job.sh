#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=exp24val
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=1-00:00
#SBATCH --output=exp24_eval_%j.out
# Powered VALIDATION only. Qwen3.5 selectors and Qwen3.8 executor run
# sequentially on one H100. The untouched test.jsonl is never opened here.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}; [ -f "$ROOT/behavior.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
REAL_HOME=$HOME
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HOME=$SCRATCH/compute_home
mkdir -p "$HOME/.cache" "$SLURM_TMPDIR/triton-selector" "$SLURM_TMPDIR/triton-executor"

DATA=experiments/results/exp24_qwen38_data/validation.jsonl
A42=$(readlink -f experiments/results/exp24_grpo_874047)
A43=$(readlink -f experiments/results/exp24_grpo_874048)
for p in "$DATA" "$A42/adapter_model.safetensors" "$A43/adapter_model.safetensors"; do
  [ -s "$p" ] || { echo "missing required artifact: $p"; exit 1; }
done
OUT=experiments/results/exp24_powered_validation
mkdir -p "$OUT"
SEL=$OUT/selections.jsonl
SCORES=$OUT/scores.jsonl
SUMMARY=$OUT/summary.json
BASE_PORT=$((10000 + ${SLURM_JOB_ID:-24} % 40000))
SELECTOR_PORT=$BASE_PORT
EXECUTOR_PORT=$((BASE_PORT + 1))
SERVER=""
cleanup() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT

run_eval_py() {
  (
    module purge
    module load StdEnv/2023
    module load gcc cuda python/3.11 python-build-bundle/2026a \
      scipy-stack/2026a arrow/19.0.1
    exec "$REAL_HOME/ENV-compress2/bin/python" "$@"
  )
}
start_vllm_stack() {
  module purge
  module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0 2>/dev/null
}
wait_server() {
  local port=$1 log=$2
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && return 0
    kill -0 "$SERVER" 2>/dev/null || { tail -100 "$log"; return 1; }
    sleep 10
  done
  echo "server on $port did not become ready"; return 1
}
stop_server() {
  [ -n "$SERVER" ] || return 0
  kill "$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
  SERVER=""
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 1500 ] && return 0
    sleep 5
  done
  echo "GPU memory did not clear after server stop"; nvidia-smi; return 1
}

# Stage 1: base plus both frozen LoRA selectors.
(
  start_vllm_stack
  exec env CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$SLURM_TMPDIR/triton-selector \
    "$REAL_HOME/ENV-vllm2/bin/vllm" serve Qwen/Qwen3.5-4B \
    --port "$SELECTOR_PORT" --served-model-name base_qwen35 \
    --max-model-len 32768 --generation-config vllm \
    --gpu-memory-utilization 0.90 --max-num-seqs 16 --max-num-batched-tokens 1024 \
    --enable-lora --max-lora-rank 16 \
    --lora-modules "grpo_seed42=$A42" "grpo_seed43=$A43"
) > experiments/vllm_exp24_eval_selector_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
wait_server "$SELECTOR_PORT" experiments/vllm_exp24_eval_selector_$SLURM_JOB_ID.log
curl -sf "http://127.0.0.1:$SELECTOR_PORT/v1/models" > "$OUT/selector_models.json"
python3 - "$OUT/selector_models.json" <<'PY'
import json,sys
ids={x['id'] for x in json.load(open(sys.argv[1]))['data']}
required={'base_qwen35','grpo_seed42','grpo_seed43'}
print('served selector models:',sorted(ids))
if not required <= ids: raise SystemExit(f'missing selector aliases: {required-ids}')
PY
run_eval_py experiments/exp24_evaluate.py select \
  --split "$DATA" --selector-url "http://127.0.0.1:$SELECTOR_PORT" \
  --out "$SEL" --workers 8
stop_server

# Stage 2: frozen Qwen3.8 executor, exact training reward interface.
(
  start_vllm_stack
  exec env CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$SLURM_TMPDIR/triton-executor \
    "$REAL_HOME/ENV-vllm2/bin/vllm" serve Qwen/Qwen3.8-27B \
    --port "$EXECUTOR_PORT" --served-model-name qwen38-exp24-eval \
    --max-model-len 32768 --generation-config vllm \
    --gpu-memory-utilization 0.92 --max-num-seqs 16 --max-num-batched-tokens 1024
) > experiments/vllm_exp24_eval_executor_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
wait_server "$EXECUTOR_PORT" experiments/vllm_exp24_eval_executor_$SLURM_JOB_ID.log
run_eval_py experiments/exp24_evaluate.py score \
  --split "$DATA" --selections "$SEL" \
  --executor-url "http://127.0.0.1:$EXECUTOR_PORT" \
  --executor-name qwen38-exp24-eval --samples 4 --max-tokens 4096 \
  --out "$SCORES" --summary "$SUMMARY" \
  --adapter42 "$A42" --adapter43 "$A43" --workers 8
stop_server
sha256sum "$DATA" "$A42/adapter_model.safetensors" \
  "$A43/adapter_model.safetensors" "$SEL" "$SCORES" "$SUMMARY" \
  > "$OUT/checksums.sha256"
echo "EXP24 POWERED VALIDATION COMPLETE -> $SUMMARY"
