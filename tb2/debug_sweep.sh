#!/bin/bash
# DEBUG SWEEP (run inside debugjob, srun -n1): validate everything cheaply
# before the big relaunch.
#   Part 1: start every one of the 89 task environments once from the baked
#           sifs (catches broken images / bootstrap issues per task).
#   Part 2: vLLM serve test at WINDOW=0.8 x native with TP as given
#           (209715 needs TP>=2 for KV; TP=2 MoE init HUNG on Narval vllm
#           0.25 - this is the make-or-break test for the window directive).
#   debugjob -g 4 srun -n1 bash tb2/debug_sweep.sh "2 4" 209715
# (debugjob only allows 1 or 4 GPUs; we test TP=2 first - schedules easier -
#  and fall back to TP=4 in the same allocation if 2 hangs like Narval)
# source Lmod init BEFORE set -u (profile scripts use unbound vars)
source /etc/profile.d/*lmod*.sh 2>/dev/null || source /etc/profile
set -u
TPS=${1:-"2 4"}
WINDOW=${2:-209715}
cd $SCRATCH/dccb

if [ "${SKIP_ENV_SWEEP:-0}" = "1" ]; then
  echo "PART 1 skipped (done separately on login node)"
else
echo "===== PART 1: env-start sweep over all baked sifs ====="
module load apptainer 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache APPTAINER_TMPDIR=${SLURM_TMPDIR:-/tmp}
ok=0; bad=0
for sif in $SCRATCH/tb2/sif_cache/*.sif; do
    name=$(basename $sif .sif)
    if timeout 120 apptainer exec "$sif" /opt/harbor-server/bin/python3 -c "import uvicorn, fastapi" >/dev/null 2>&1; then
        ok=$((ok+1))
    else
        bad=$((bad+1)); echo "  BAD: $name"
    fi
done
echo "env sweep: $ok ok, $bad bad"
fi

export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0 2>/dev/null
export HOME=$SCRATCH/compute_home; mkdir -p $HOME/.cache
source /home/anangia/ENV-vllm2/bin/activate
for TP in $TPS; do
  echo "===== PART 2: vLLM TP=$TP serve @ window=$WINDOW ====="
  LOG=$SCRATCH/debug_serve_tp$TP.log
  vllm serve Qwen/Qwen3.5-35B-A3B --port 8000 --served-model-name dbg \
      --tensor-parallel-size $TP --max-model-len $WINDOW \
      --max-num-seqs 8 --max-num-batched-tokens 1024 \
      --gpu-memory-utilization 0.92 > $LOG 2>&1 &
  PID=$!
  UP=false
  for i in $(seq 1 90); do   # 15 min: the Narval hang manifested as init stall
      curl -s http://127.0.0.1:8000/health >/dev/null && { echo "TP=$TP @ $WINDOW: UP after ${i}0s"; UP=true; break; }
      kill -0 $PID 2>/dev/null || { echo "TP=$TP @ $WINDOW: DIED"; tail -5 $LOG; break; }
      sleep 10
  done
  if $UP; then
      curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
          -d '{"model":"dbg","messages":[{"role":"user","content":"say OK"}],"max_tokens":16}' \
          | python3 -c "import sys,json; print('completion:', json.load(sys.stdin)['choices'][0]['message']['content'][:60])"
      kill $PID 2>/dev/null; sleep 10
      echo "PART 2 PASS at TP=$TP"; exit 0
  fi
  echo "TP=$TP failed/hung - killing and trying next"
  kill -9 $PID 2>/dev/null; pkill -9 -f 'vllm serve' 2>/dev/null; sleep 20
done
echo "PART 2 FAIL: no TP configuration served window=$WINDOW"; exit 3
