#!/bin/bash
# LOGIN NODE: the on-policy glue (group decision: NO other-model traces).
# 1. Wait for the 35B TB2 jobs (easy25 + 3 shards) to leave the queue.
# 2. Build on-policy example sets from OUR trajectories with the 35B tokenizer:
#    4k contexts (n=64) and 16k contexts (n=64, needs long trajectories).
# 3. Queue the behavioral suite on those files only (run_all.sh queue-onpolicy).
#   setsid nohup bash tb2/onpolicy_watcher.sh > $SCRATCH/onpolicy_watcher.log 2>&1 &
set -u
cd $SCRATCH/dccb
JOBS=$(cat $SCRATCH/tb2_jobs.txt)   # written by tb2/relaunch_orchestrator.sh

while true; do
    left=0
    for j in $JOBS; do squeue -j $j -h 2>/dev/null | grep -q . && left=$((left+1)); done
    [ $left -eq 0 ] && break
    echo "[$(date +%m-%d\ %H:%M)] $left TB2 jobs still queued/running"
    sleep 600
done
echo "[$(date +%m-%d\ %H:%M)] all TB2 jobs done; building on-policy examples"

module load gcc cuda python/3.11 arrow/19.0.1 2>/dev/null
source $HOME/ENV-compress2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1

GLOB="$SCRATCH/tb2/jobs/tb2-qwen35-35b-bf16*/*/agent/trajectory.json"
ls $GLOB >/dev/null 2>&1 || { echo "FATAL: no trajectories matched $GLOB"; exit 1; }
echo "trajectories: $(ls $GLOB | wc -l)"

# ONE data policy (group directive): each example keeps its FULL real history,
# capped at 0.8 x the model's native window (read from its own config).
# The old 4k/16k fixed-size tiers are DISCARDED.
CTX=$(python - <<'PY'
import json, glob
c=json.load(open(glob.glob(f"{__import__('os').environ['SCRATCH']}/hf/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/*/config.json")[0]))
m=c.get('max_position_embeddings') or c.get('text_config',{}).get('max_position_embeddings')
print(int(m*0.8))
PY
)
echo "context cap = $CTX tokens (0.8 x native)"
python prefetch_onpolicy.py --model Qwen/Qwen3.5-35B-A3B \
    --traj-glob "$GLOB" --context-tokens $CTX --min-context-tokens 2048 \
    --recent-tokens 4096 --num-examples 64 --max-per-task 2 \
    --out data/examples_onpolicy.json
n=$(python -c "import json;print(len(json.load(open('data/examples_onpolicy.json'))))" 2>/dev/null || echo 0)
echo "data/examples_onpolicy.json: $n examples (full-context, on-policy)"
[ "$n" -lt 2 ] && { echo "FATAL: not enough on-policy examples"; exit 1; }

( cd experiments && SBATCH_ACCOUNT=def-rgrosse bash run_all.sh queue-onpolicy )
echo "[$(date +%m-%d\ %H:%M)] on-policy suite submitted"
