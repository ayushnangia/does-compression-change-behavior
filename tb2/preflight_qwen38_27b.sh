#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=q38preflight
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-01:00
#SBATCH --output=q38_preflight_%j.out
# Architecture + generation + parser gate for the final Qwen3.8-27B lineage.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}; [ -f "$ROOT/behavior.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
module load gcc cuda python/3.12 arrow/19.0.1 2>/dev/null
REAL_HOME=$HOME; export HOME=$SCRATCH/compute_home HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
mkdir -p "$HOME/.cache" "$SLURM_TMPDIR/triton"
TRITON_CACHE_DIR=$SLURM_TMPDIR/triton $REAL_HOME/ENV-vllm2/bin/vllm serve \
  Qwen/Qwen3.8-27B --port 8000 --served-model-name qwen38-27b-preflight \
  --max-model-len 77824 --generation-config vllm --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 --max-num-batched-tokens 1024 > q38_vllm_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { tail -100 q38_vllm_$SLURM_JOB_ID.log; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null || exit 1

# Use a real serialized decision point, not a toy chat prompt. Require at least
# one parser-recognized action across four deployment-sampling draws.
module unload python 2>/dev/null || true
module load python/3.11 2>/dev/null
$REAL_HOME/ENV-compress2/bin/python - <<'PY'
import json, requests
from transformers import AutoTokenizer
from behavior import parse_action
x=json.load(open('data/examples_onpolicy.json'))
t=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B',trust_remote_code=True)
prompt=t.decode(x[5]['context_ids'][-28000:],skip_special_tokens=False)
r=requests.post('http://127.0.0.1:8000/v1/completions',json={
 'model':'qwen38-27b-preflight','prompt':prompt,'n':4,'max_tokens':2048,
 'temperature':1.0,'top_p':1.0,'seed':38},timeout=900)
r.raise_for_status(); texts=[c['text'] for c in r.json()['choices']]
a=[parse_action(s) for s in texts]
print('parsed actions:',a)
assert any(v is not None for v in a), 'Qwen3.8 generated no parser-recognized action'
print('QWEN38 PREFLIGHT GREEN')
PY
