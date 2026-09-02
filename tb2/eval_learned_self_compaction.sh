#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=q38selfcompact
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=0-12:00
#SBATCH --output=q38_self_compact_%j.out
# Closed-loop learned compaction: frozen Qwen3.8 agent on GPU0 delegates
# selection over its own live history to a frozen exp24 Qwen3.5 LoRA on GPU1.
#
# Usage (only after held-out selector comparison freezes one adapter):
#   sbatch tb2/eval_learned_self_compaction.sh ADAPTER_DIR SEED [SUBSET]
set -euo pipefail
ADAPTER=${1:?usage: eval_learned_self_compaction.sh ADAPTER_DIR SEED [SUBSET]}
SEED=${2:?seed label required}
SUBSET=${3:-easy25}
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}; [ -f "$ROOT/behavior.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
ADAPTER=$(readlink -f "$ADAPTER")
[ -s "$ADAPTER/adapter_config.json" ] || { echo "missing adapter config: $ADAPTER"; exit 1; }
[ -s "$ADAPTER/adapter_model.safetensors" ] || { echo "missing adapter weights: $ADAPTER"; exit 1; }

REAL_HOME=$HOME
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1
COMPUTE_HOME=$SCRATCH/compute_home
mkdir -p "$COMPUTE_HOME/.cache" "$SLURM_TMPDIR/triton-selector"
BASE_PORT=$((10000 + ${SLURM_JOB_ID:-24} % 40000))
SELECTOR_PORT=$((BASE_PORT + 1))
SELECTOR_NAME=qwen35-exp24-selector
export EXP24_SELECTOR_API_BASE=http://127.0.0.1:$SELECTOR_PORT/v1
export EXP24_SELECTOR_MODEL=$SELECTOR_NAME
export EXP24_SELECTOR_LOG=$ROOT/experiments/results/exp24_live_selector_${SLURM_JOB_ID}.jsonl
export TB2_AGENT_IMPORT_PATH=exp22.compaction_agents:LearnedSelectorTerminus

# Immutable deployment provenance before any task starts.
python3 - "$ADAPTER" "$SEED" "$EXP24_SELECTOR_LOG" <<'PY'
import hashlib,json,sys
from pathlib import Path
adapter=Path(sys.argv[1]); out=Path(sys.argv[3]).with_suffix('.manifest.json')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir() if p.is_file()}
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"selector_base":"Qwen/Qwen3.5-4B","adapter":str(adapter),
 "seed":sys.argv[2],"adapter_sha256":files,"acting_model":"Qwen/Qwen3.8-27B",
 "agent":"exp22.compaction_agents:LearnedSelectorTerminus",
 "disposition":"closed-loop on-policy outcome evaluation"},indent=2)+"\n")
PY

# Selector endpoint: Qwen3.5 base plus exactly one frozen LoRA adapter.
module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0 2>/dev/null
(
  exec env HOME=$COMPUTE_HOME CUDA_VISIBLE_DEVICES=1 \
    TRITON_CACHE_DIR=$SLURM_TMPDIR/triton-selector \
    $REAL_HOME/ENV-vllm2/bin/vllm serve Qwen/Qwen3.5-4B \
    --port "$SELECTOR_PORT" --max-model-len 32768 --generation-config vllm \
    --gpu-memory-utilization 0.80 --max-num-seqs 8 --max-num-batched-tokens 1024 \
    --enable-lora --max-lora-rank 16 \
    --lora-modules "$SELECTOR_NAME=$ADAPTER"
) > experiments/vllm_selector_$SLURM_JOB_ID.log 2>&1 &
SELECTOR_PID=$!
trap 'kill $SELECTOR_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$SELECTOR_PORT/health >/dev/null && break
  kill -0 $SELECTOR_PID 2>/dev/null || {
    tail -100 experiments/vllm_selector_$SLURM_JOB_ID.log; exit 1;
  }
  sleep 10
done
curl -sf http://127.0.0.1:$SELECTOR_PORT/health >/dev/null || {
  echo selector-not-ready; exit 1;
}
# Competence gate: deployed endpoint must emit strict, in-range keep JSON.
python3 - <<'PY'
import json,os,urllib.request
from experiments.exp24_data import parse_keep
payload={"model":os.environ["EXP24_SELECTOR_MODEL"],"messages":[{"role":"user",
 "content":'Select old blocks. Max copied characters: 10. [0] chars=1 A '
           '[1] chars=1 B. Return JSON only now, exactly: {"keep":[0,3,...]}'}],
 "temperature":0,"max_tokens":64,
 "chat_template_kwargs":{"enable_thinking":False}}
req=urllib.request.Request(os.environ["EXP24_SELECTOR_API_BASE"]+"/chat/completions",
 data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req,timeout=180) as r: x=json.load(r)
text=x['choices'][0]['message'].get('content') or ''
keep,valid=parse_keep(text,2)
print('deployed selector gate:',repr(text),'parsed=',keep,'valid=',valid)
if not valid: raise SystemExit('deployed selector emitted invalid JSON')
PY

# Reuse the audited offline Harbor runner for the acting model. Restricting
# CUDA visibility prevents its vLLM from touching the selector GPU.
CUDA_VISIBLE_DEVICES=0 bash tb2/eval_tb2.sh Qwen/Qwen3.8-27B \
  "qwen38-selfcompact-s${SEED}" 1 "$SUBSET" 4 65536 1 "$BASE_PORT"
