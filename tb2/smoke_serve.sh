#!/bin/bash
# Smoke-test gate: serve any cached model with vLLM on one node, demand one
# completion, exit 0/1. Chain the real eval behind it with --dependency=afterok.
#
# Usage (Trillium; CLI flags override the Narval header per Slurm rules):
#   sbatch --gpus-per-node=h100:1 --time=1:00:00 \
#       --export=ALL,VLLM_EXTRA_ARGS="--max-num-seqs 128 --max-num-batched-tokens 1024",GPU_UTIL=0.92 \
#       tb2/smoke_serve.sh Qwen/Qwen3.5-35B-A3B qwen35-35b-bf16
# Or interactively the moment nodes return:
#   debugjob -g 1 srun bash tb2/smoke_serve.sh Qwen/Qwen3.5-35B-A3B qwen35-35b-bf16
#
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=1:00:00
#SBATCH --output=smoke_%j.out
set -u

MODEL=${1:?usage: smoke_serve.sh <hf-model-id> <served-name> [tp]}
SERVED=${2:?served-model-name}
TP=${3:-1}
PORT=8000

if [[ $(hostname) == trig* ]]; then
    VLLM_ENV=${VLLM_ENV:-$HOME/ENV-vllm2}
    module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0 2>/dev/null  # gcc BEFORE cuda
else
    VLLM_ENV=${VLLM_ENV:-$SCRATCH/ENV-vllm2}
    module load cuda/12.9 opencv python/3.12 2>/dev/null
fi
GPU_UTIL=${GPU_UTIL:-0.90}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
cd $SCRATCH

echo "[smoke] node=$(hostname) model=$MODEL tp=$TP util=$GPU_UTIL extra='$VLLM_EXTRA_ARGS'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$VLLM_ENV/bin/activate"
LOG=$SCRATCH/smoke_vllm_${SLURM_JOB_ID:-manual}.log
# NO stdbuf: LD_PRELOAD leaks into nvcc's /bin/sh, kills flashinfer JIT (GLIBC_ABI_DT_RELR)
vllm serve "$MODEL" --port $PORT --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP" --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_UTIL" $VLLM_EXTRA_ARGS > "$LOG" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 120); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null && { echo "[smoke] vLLM up after ${i}0s"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "[smoke] vLLM DIED; last log:"; tail -20 "$LOG"; exit 1; }
    sleep 10
done
curl -s "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[smoke] never came up"; tail -20 "$LOG"; kill $VLLM_PID; exit 1; }

echo "[smoke] requesting completion ..."
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: SMOKE OK\"}],\"max_tokens\":64,\"temperature\":0}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('[smoke] reply:', r['choices'][0]['message']['content'].strip()[:100])"
RC=$?

nvidia-smi --query-gpu=memory.used --format=csv,noheader
kill $VLLM_PID 2>/dev/null
echo "[smoke] exit=$RC"
exit $RC
