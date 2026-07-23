#!/bin/bash
# Trillium debugjob smoke test: serve bf16 Qwen3.5-35B-A3B on ONE H100-80GB
# and get a completion out of it. This is the run Narval could not do.
#   debugjob -g 1 srun bash $SCRATCH/dccb/experiments/smoke_35b_trillium.sh
set -u
echo "[smoke] node=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
cd $SCRATCH

module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0
source $HOME/ENV-vllm2/bin/activate

# NOTE: no stdbuf here. Gentoo's libstdbuf.so gets LD_PRELOADed into nvcc's
# /bin/sh (system glibc), killing flashinfer JIT builds with
# "GLIBC_ABI_DT_RELR not found". PYTHONUNBUFFERED=1 covers buffering.
LOG=$SCRATCH/smoke_35b_vllm.log
vllm serve Qwen/Qwen3.5-35B-A3B --port 8000 \
    --served-model-name qwen35-35b-bf16 \
    --tensor-parallel-size 1 --max-model-len 32768 \
    --max-num-batched-tokens 1024 --max-num-seqs 128 \
    --gpu-memory-utilization 0.92 > $LOG 2>&1 &
# --max-num-seqs 128: Qwen3.5 linear-attn needs one Mamba cache block per
# decode seq; only ~135 blocks fit beside 66GB of weights on one H100-80.
VLLM_PID=$!

for i in $(seq 1 120); do
    curl -s http://127.0.0.1:8000/health > /dev/null && { echo "[smoke] vLLM up after ${i}0s"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "[smoke] vLLM DIED; last log:"; tail -20 $LOG; exit 1; }
    sleep 10
done
curl -s http://127.0.0.1:8000/health > /dev/null || { echo "[smoke] never came up"; tail -20 $LOG; kill $VLLM_PID; exit 1; }

echo "[smoke] requesting completion ..."
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen35-35b-bf16","messages":[{"role":"user","content":"Reply with exactly: SMOKE OK"}],"max_tokens":32,"temperature":0}' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('[smoke] reply:', r['choices'][0]['message']['content'].strip())"
RC=$?

nvidia-smi --query-gpu=memory.used --format=csv,noheader
kill $VLLM_PID 2>/dev/null
echo "[smoke] exit=$RC"
exit $RC
