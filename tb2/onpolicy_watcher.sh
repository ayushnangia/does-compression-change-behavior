#!/bin/bash
# LOGIN NODE: the on-policy glue (group decision: NO other-model traces).
# 1. Wait for the 35B TB2 jobs (easy25 + 3 shards) to leave the queue.
# 2. Build on-policy example sets from OUR trajectories with the 35B tokenizer:
#    4k contexts (n=64) and 16k contexts (n=64, needs long trajectories).
# 3. Queue the behavioral suite on those files only (run_all.sh queue-onpolicy).
#   setsid nohup bash tb2/onpolicy_watcher.sh > $SCRATCH/onpolicy_watcher.log 2>&1 &
set -u
cd $SCRATCH/dccb
JOBS="694202 694275 694276 694277"

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

python prefetch_onpolicy.py --model Qwen/Qwen3.5-35B-A3B \
    --traj-glob "$GLOB" --context-tokens 4096 --recent-tokens 512 \
    --num-examples 64 --max-per-task 2 --out data/examples_onpolicy_4k.json
python prefetch_onpolicy.py --model Qwen/Qwen3.5-35B-A3B \
    --traj-glob "$GLOB" --context-tokens 16384 --recent-tokens 1024 \
    --num-examples 64 --max-per-task 2 --out data/examples_onpolicy_16k.json

for f in data/examples_onpolicy_4k.json data/examples_onpolicy_16k.json; do
    n=$(python -c "import json;print(len(json.load(open('$f'))))" 2>/dev/null || echo 0)
    echo "$f: $n examples"
    [ "$n" -lt 16 ] && { echo "  too few - removing so run_all skips its exps"; rm -f "$f"; }
done

( cd experiments && SBATCH_ACCOUNT=def-rgrosse bash run_all.sh queue-onpolicy )
echo "[$(date +%m-%d\ %H:%M)] on-policy suite submitted"
