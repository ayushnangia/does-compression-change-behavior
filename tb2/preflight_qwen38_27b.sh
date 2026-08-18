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

# Use a real serialized decision point, not a toy chat prompt. Decode it with
# the tokenizer that created those IDs, then parse RAW model responses with
# Harbor's live authority parser (not behavior.parse_action, which parses the
# stored <tool_calls> serialization after Harbor has already acted).
module unload python 2>/dev/null || true
module load python/3.11 2>/dev/null
$REAL_HOME/ENV-compress2/bin/python - <<'PY'
import json, os
from transformers import AutoTokenizer
x=json.load(open('data/examples_onpolicy.json'))
t=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B',trust_remote_code=True)
open(os.path.join(os.environ['SLURM_TMPDIR'],'q38_prompt.txt'),'w').write(
    t.decode(x[5]['context_ids'][-28000:],skip_special_tokens=False))
PY
module unload python 2>/dev/null || true
module load python/3.12 2>/dev/null
$REAL_HOME/ENV-harbor2/bin/python - <<'PY'
import os, requests
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
prompt=open(os.path.join(os.environ['SLURM_TMPDIR'],'q38_prompt.txt')).read()
r=requests.post('http://127.0.0.1:8000/v1/completions',json={
 'model':'qwen38-27b-preflight','prompt':prompt,'n':2,'max_tokens':4096,
 'temperature':1.0,'top_p':1.0,'seed':38},timeout=1200)
r.raise_for_status(); texts=[c['text'] for c in r.json()['choices']]
for i,text in enumerate(texts):
    print(f'--- raw completion {i} ---\n{text[:2000]}\n--- end snippet ---')
parsed=[TerminusJSONPlainParser().parse_response(text) for text in texts]
counts=[len(p.commands) for p in parsed]
print('authority-parser command counts:',counts)
assert any(n > 0 for n in counts), 'Qwen3.8 generated no Harbor-parsed command'
print('QWEN38 PREFLIGHT GREEN')
PY
