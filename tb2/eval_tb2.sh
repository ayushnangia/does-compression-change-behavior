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
# Narval header below; on H100 clusters edit gres per MIGRATION.md.
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --output=tb2_%j.out
set -u

MODEL=${1:?usage: eval_tb2.sh <hf-model-id> <served-name> [tp] [easy25]}
SERVED=${2:?served-model-name (exactly one / rule: hosted_vllm/<served-name>)}
TP=${3:-1}
SUBSET=${4:-}
PORT=8000
VLLM_ENV=${VLLM_ENV:-$SCRATCH/ENV-vllm2}
HARBOR_ENV=${HARBOR_ENV:-$SCRATCH/ENV-harbor2}
TB2_DIR=${TB2_DIR:-$SCRATCH/tb2}
HERE=$(cd "$(dirname "$0")" && pwd)

# ---- offline etiquette: nothing here may touch the internet ----
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export LITELLM_LOCAL_MODEL_COST_MAP=True   # stop litellm fetching pricing json

# ---- 1. vLLM in the background ----
module load cuda/12.9 opencv python/3.12 2>/dev/null
source "$VLLM_ENV/bin/activate"
vllm serve "$MODEL" --port $PORT --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP" --max-model-len 32768 \
    --gpu-memory-utilization 0.90 > "vllm_$SLURM_JOB_ID.log" 2>&1 &
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
sed -e "s|@SERVED@|$SERVED|g" -e "s|@TB2@|$TB2_DIR|g" -e "s|@PORT@|$PORT|g" \
    "$HERE/config_template.yaml" > "$CONFIG"
if [ "$SUBSET" = "easy25" ]; then
    { echo "tasks:"; sed 's/^/  - /' "$HERE/easy25.txt"; } >> "$CONFIG"
fi

# ---- 3. harbor (terminus-2, apptainer from the sif cache) ----
module load apptainer gcc arrow 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache APPTAINER_TMPDIR=$SLURM_TMPDIR
source "$HARBOR_ENV/bin/activate"
harbor run -c "$CONFIG" -y
STATUS=$?

kill $VLLM_PID 2>/dev/null
exit $STATUS
