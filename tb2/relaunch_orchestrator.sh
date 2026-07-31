#!/bin/bash
# LOGIN NODE, detached: the full test-then-relaunch sequence (group directive:
# debug + test first, then the real runs; window = 0.8 x native from config).
#   1. Import-test every baked sif once (login node, cheap).
#   2. Debugjob serve gate: TP=2 then TP=4 at window 209715 (Narval hang risk).
#   3. On PASS: relaunch TB2 easy25 + 3 shards at the working TP + window,
#      write job ids for the on-policy watcher, restart the watcher.
#   setsid nohup bash tb2/relaunch_orchestrator.sh > $SCRATCH/orchestrator.log 2>&1 &
set -u
cd $SCRATCH/dccb
WINDOW=209715

echo "===== 1. sif import sweep (all 89, login node) ====="
module load apptainer 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache APPTAINER_TMPDIR=/tmp/$USER-apt
mkdir -p $APPTAINER_TMPDIR
ok=0; bad=0
for sif in $SCRATCH/tb2/sif_cache/*.sif; do
    if timeout 120 apptainer exec "$sif" /opt/harbor-server/bin/python3 -c "import uvicorn, fastapi" >/dev/null 2>&1; then
        ok=$((ok+1))
    else
        bad=$((bad+1)); echo "  BAD SIF: $(basename $sif)"
    fi
done
echo "sif sweep: $ok ok, $bad bad"
[ $bad -gt 0 ] && echo "WARNING: rebake the BAD ones before their tasks run"

echo "===== 2. serve gate: TP sweep @ window=$WINDOW (debugjob) ====="
GATE_LOG=$SCRATCH/debug_sweep_gate.log
SKIP_ENV_SWEEP=1 timeout 6500 debugjob -g 4 --account=def-rgrosse srun -n1 \
    bash tb2/debug_sweep.sh 2,4 $WINDOW > $GATE_LOG 2>&1
TP=$(grep -oP "PART 2 PASS at TP=\K\d+" $GATE_LOG || true)
if [ -z "$TP" ]; then
    echo "FATAL: no TP served window=$WINDOW - human decision needed (see $GATE_LOG)"
    tail -5 $GATE_LOG
    exit 1
fi
echo "serve gate PASSED at TP=$TP"

echo "===== 3. relaunch TB2 at TP=$TP window=$WINDOW ====="
export SBATCH_ACCOUNT=def-rgrosse
JOBS=""
for SUB in easy25 tb2/shard_00.txt tb2/shard_01.txt tb2/shard_02.txt; do
    [ -f "$SUB" ] && SUBARG=$SCRATCH/dccb/$SUB || SUBARG=$SUB
    J=$(sbatch --parsable --gpus-per-node=h100:$TP --time=1-00:00 \
        tb2/eval_tb2.sh Qwen/Qwen3.5-35B-A3B qwen35-35b-bf16 $TP "$SUBARG" 4 $WINDOW) \
        && JOBS="$JOBS $J" && echo "  $SUB -> $J"
done
echo "$JOBS" | xargs > $SCRATCH/tb2_jobs.txt
echo "job ids written: $(cat $SCRATCH/tb2_jobs.txt)"

pkill -f onpolicy_watcher.sh 2>/dev/null; sleep 2
setsid nohup bash tb2/onpolicy_watcher.sh > $SCRATCH/onpolicy_watcher.log 2>&1 &
echo "watcher restarted; orchestration complete"
