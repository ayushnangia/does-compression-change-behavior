#!/bin/bash
# DEBUGJOB, ONE WINDOW PER JOB: fresh GPUs guaranteed by job isolation, so a
# crashed attempt can never poison the next one (the 21GB ghost lesson).
#   debugjob -g 4 --account=def-rgrosse srun -n1 bash tb2/gate_try_window.sh <window>
source /etc/profile.d/*lmod*.sh 2>/dev/null || source /etc/profile
set -u
WINDOW=${1:?window}
TP=4
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0 2>/dev/null
export HOME=$SCRATCH/compute_home; mkdir -p $HOME/.cache
source /home/anangia/ENV-vllm2/bin/activate
cd $SCRATCH

# wedged-GPU guard (job should start clean; verify)
GHOST=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
[ "${GHOST:-0}" -gt 2000 ] && { echo "WEDGED node $(hostname), no verdict"; exit 99; }

LOG=$SCRATCH/gatetry_w$WINDOW.log
vllm serve Qwen/Qwen3.5-35B-A3B --port 8000 --served-model-name dbg \
    --tensor-parallel-size $TP --max-model-len $WINDOW \
    --max-num-seqs 128 --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.92 > $LOG 2>&1 &
PID=$!
for i in $(seq 1 90); do
    curl -s http://127.0.0.1:8000/health >/dev/null && {
        R=$(curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
            -d '{"model":"dbg","messages":[{"role":"user","content":"say OK"}],"max_tokens":8}' \
            | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:20])" 2>/dev/null)
        if [ -n "$R" ]; then
            echo "TP=$TP WINDOW=$WINDOW" > $SCRATCH/gate_verdict.txt
            echo "PASS window=$WINDOW (reply: $R)"; kill $PID; exit 0
        fi
    }
    kill -0 $PID 2>/dev/null || break
    sleep 10
done
echo "$WINDOW" >> $SCRATCH/gate_failed_windows.txt
echo "FAIL window=$WINDOW"; tail -3 $LOG | cut -c1-140
exit 1
