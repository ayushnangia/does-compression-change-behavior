#!/bin/bash
# Terminal-Bench 2.0 evaluation, fully offline on one GPU node.
# vLLM serves the model on :8000; harbor runs Terminus 2 inside Apptainer
# containers from the pre-baked .sif cache. One script for every model.
#
# Usage:
#   sbatch eval_tb2.sh Qwen/Qwen3.5-9B qwen35-9b
#   sbatch eval_tb2.sh zai-org/GLM-4.7-Flash glm47-flash 2      # TP=2
#   sbatch --time=0-2:00 eval_tb2.sh Qwen/Qwen3.5-9B qwen35-9b 1 easy25
#
# Prereqs (login node, once):
#   1. model cached:   HF_HOME=$SCRATCH/hf hf download <model>
#   2. tasks:          $SCRATCH/tb2/terminal-bench (harbor dataset download)
#   3. images baked:   bash bake_all_sifs.sh
#
# Cluster-neutral header (Trillium REJECTS --mem/--cpus directives; per-gpu
# jobs there always get 186 GiB host memory). Override per cluster at submit:
#   Trillium: sbatch --gpus-per-node=h100:1 tb2/eval_tb2.sh ...
#   Narval:   sbatch --mem=64G --cpus-per-task=12 tb2/eval_tb2.sh ...
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=0-12:00
#SBATCH --output=tb2_%j.out
set -u

MODEL=${1:?usage: eval_tb2.sh <hf-model-id> <served-name> [tp] [easy25]}
SERVED=${2:?served-model-name (exactly one / rule: hosted_vllm/<served-name>)}
TP=${3:-1}
SUBSET=${4:-}
PORT=8000
# Under sbatch, $0 is Slurm's spooled COPY in /var/spool/slurm/... - the
# sidecar files (config_template.yaml, easy25.txt) are not next to it.
# Resolve the real tb2/ dir via fallbacks (job 679008 died on this).
HERE=$(cd "$(dirname "$0")" && pwd)
if [ ! -f "$HERE/config_template.yaml" ]; then
    for c in "${SLURM_SUBMIT_DIR:-.}/tb2" "${SLURM_SUBMIT_DIR:-.}" "$SCRATCH/dccb/tb2"; do
        [ -f "$c/config_template.yaml" ] && HERE=$(cd "$c" && pwd) && break
    done
fi
[ -f "$HERE/config_template.yaml" ] || { echo "cannot locate tb2/ sidecar files"; exit 1; }

# ---- cluster detection (Trillium vs Narval; see docs/MIGRATION.md) ----
if [[ $(hostname) == trig* ]]; then
    ON_TRILLIUM=true
    VLLM_ENV=${VLLM_ENV:-$HOME/ENV-vllm2}      # venvs live in $HOME on Trillium
    HARBOR_ENV=${HARBOR_ENV:-$HOME/ENV-harbor2}
else
    ON_TRILLIUM=false
    VLLM_ENV=${VLLM_ENV:-$SCRATCH/ENV-vllm3}
    HARBOR_ENV=${HARBOR_ENV:-$SCRATCH/ENV-harbor2}
fi
TB2_DIR=${TB2_DIR:-$SCRATCH/tb2}
# Trillium's sbatch wrapper forces --export=NONE: env vars DO NOT reach the
# job. Per-model serving requirements are therefore baked in here.
case "$MODEL" in
  *Qwen3.5-35B*)  # Mamba cache: one block per decode seq; ~135 fit at util .92
    GPU_UTIL=${GPU_UTIL:-0.92}
    VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:---max-num-seqs 128 --max-num-batched-tokens 1024} ;;
esac
GPU_UTIL=${GPU_UTIL:-0.90}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

# ---- wedged-GPU guard: a prior CUDA illegal-access can leave GPU memory
# pinned with no owning process (trig0013, Jul 31 - 21GB ghost); any work on
# such a node is garbage. Fail loud and fast instead. ----
GHOST=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if [ "${GHOST:-0}" -gt 2000 ] && ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "FATAL WEDGED_GPU: ${GHOST}MB pinned with no process on $(hostname) - needs driver reset; resubmit with --exclude=$(hostname -s)"
    exit 99
fi

# ---- offline etiquette: nothing here may touch the internet ----
# Trillium compute nodes mount $HOME READ-ONLY: redirect HOME to a writable
# scratch home for ~/.cache writers (torch.compile, flashinfer, triton, harbor).
# Venv paths above already resolved against the real home.
if $ON_TRILLIUM; then export HOME=$SCRATCH/compute_home; mkdir -p $HOME/.cache; fi
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export LITELLM_LOCAL_MODEL_COST_MAP=True   # stop litellm fetching pricing json

# ---- 1. vLLM in the background ----
# NO stdbuf around vllm: its LD_PRELOAD leaks into nvcc's /bin/sh and kills
# flashinfer JIT builds (GLIBC_ABI_DT_RELR); PYTHONUNBUFFERED covers buffering.
export PYTHONUNBUFFERED=1
if $ON_TRILLIUM; then
    module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0 2>/dev/null  # gcc BEFORE cuda
else
    module load cuda/12.9 opencv python/3.12 2>/dev/null
fi
source "$VLLM_ENV/bin/activate"
# WINDOW: the model's NATIVE context minus 20% (group directive), read from
# its own config.json - override with WINDOW=<n> arg6 if KV won't fit at this TP.
WINDOW=${6:-$(python3 - <<PY
import json, glob, os
p = glob.glob(os.environ['HF_HOME'] + '/hub/models--' + '$MODEL'.replace('/','--') + '/snapshots/*/config.json')[0]
c = json.load(open(p))
m = c.get('max_position_embeddings') or c.get('text_config',{}).get('max_position_embeddings')
print(int(m*0.8))
PY
)}
echo "serving window: $WINDOW (0.8 x native unless overridden)"
vllm serve "$MODEL" --port $PORT --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP" --max-model-len $WINDOW \
    --gpu-memory-utilization "$GPU_UTIL" $VLLM_EXTRA_ARGS > "vllm_$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
deactivate

for i in $(seq 1 180); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null && { echo "vLLM up after ${i}0s"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died - see vllm_$SLURM_JOB_ID.log"; exit 1; }
    sleep 10
done
curl -s "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM never came up"; exit 1; }

# ---- 2. write the harbor config for this model ----
CONFIG=$SLURM_TMPDIR/job_config.yaml
# input cap must leave room for max_output inside max-model-len, or requests
# near the window get rejected by vLLM (input + 10240 out > window)
INPUT_CAP=$((WINDOW - 12288))
sed -e "s|@SERVED@|$SERVED|g" -e "s|@TB2@|$TB2_DIR|g" -e "s|@PORT@|$PORT|g" \
    -e "s|max_input_tokens: .*|max_input_tokens: $INPUT_CAP|" \
    "$HERE/config_template.yaml" > "$CONFIG"
# SUBSET: 'easy25' or a path to any task-list file (one task name per line)
SUBSET_FILE=""
[ "$SUBSET" = "easy25" ] && SUBSET_FILE="$HERE/easy25.txt"
[ -n "$SUBSET" ] && [ -f "$SUBSET" ] && SUBSET_FILE="$SUBSET"
if [ -n "$SUBSET_FILE" ]; then
    # harbor 0.20 wants dict entries (- path: ...), not bare names, and the
    # datasets: block must go or the full 89 run alongside the subset
    sed -i '/^datasets:/,+1d' "$CONFIG"
    sed -i "s/^job_name: .*/job_name: tb2-$SERVED-$(basename $SUBSET_FILE .txt)/" "$CONFIG"
    { echo "tasks:"; sed "s|^|  - path: $TB2_DIR/terminal-bench/|" "$SUBSET_FILE"; } >> "$CONFIG"
fi

# ---- 3. harbor (terminus-2, apptainer from the sif cache) ----
module load apptainer gcc arrow 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache APPTAINER_TMPDIR=$SLURM_TMPDIR
source "$HARBOR_ENV/bin/activate"
# --agent-timeout-multiplier: thinking models burn wall-clock; the Narval
# runs used 4x and dropping it produced a wall of AgentTimeoutError (683764)
# env-build multiplier 4: with vLLM + 2 concurrent container builds on one
# node, servers start fine but blow harbor's default handshake budget
# (9/25 GLM trials died as EnvironmentStartTimeout while uvicorn was up)
harbor run -c "$CONFIG" --agent-timeout-multiplier "${5:-4}" \
    --environment-build-timeout-multiplier 4 -y
STATUS=$?

kill $VLLM_PID 2>/dev/null
exit $STATUS
