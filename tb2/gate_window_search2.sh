#!/bin/bash
# DEBUGJOB: find the LARGEST window vllm 0.25 can actually serve for the 35B.
# 209715 (0.8 x native) faults with CUDA illegal memory access at TP=2; TP=4
# was contaminated by leftover GPU memory. Test TP=4 descending windows with
# PROPER cleanup (poll nvidia-smi until GPUs are actually free).
# Writes verdict to $SCRATCH/gate_verdict.txt as: TP=<n> WINDOW=<w>
#   debugjob -g 4 --account=def-rgrosse srun -n1 bash tb2/gate_window_search.sh
source /etc/profile.d/*lmod*.sh 2>/dev/null || source /etc/profile
set -u
TP=4
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
module load gcc cuda python/3.12 arrow/19.0.1 opencv/4.13.0 2>/dev/null
export HOME=$SCRATCH/compute_home; mkdir -p $HOME/.cache
source /home/anangia/ENV-vllm2/bin/activate
cd $SCRATCH

wait_gpus_free() {
    for i in $(seq 1 20); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
        [ "$used" -lt 2000 ] && return 0
        sleep 10
    done
    echo "WARN: GPUs never freed (max used ${used}MB)"; return 1
}

FAILED_LIST=$SCRATCH/gate_failed_windows.txt; touch $FAILED_LIST
for WINDOW in 209715 163840 131072 98304 65536; do
    grep -qx "$WINDOW" $FAILED_LIST && { echo "skip $WINDOW (known failed)"; continue; }
    echo "=== try TP=$TP window=$WINDOW ==="
    wait_gpus_free
    LOG=$SCRATCH/gatesearch_w$WINDOW.log
    vllm serve Qwen/Qwen3.5-35B-A3B --port 8000 --served-model-name dbg \
        --tensor-parallel-size $TP --max-model-len $WINDOW \
        --max-num-seqs 128 --max-num-batched-tokens 1024 \
        --gpu-memory-utilization 0.92 > $LOG 2>&1 &
    PID=$!
    UP=false
    for i in $(seq 1 60); do
        curl -s http://127.0.0.1:8000/health >/dev/null && { UP=true; break; }
        kill -0 $PID 2>/dev/null || break
        sleep 10
    done
    if $UP; then
        # prove it with a completion before declaring victory
        R=$(curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
            -d '{"model":"dbg","messages":[{"role":"user","content":"say OK"}],"max_tokens":8}' \
            | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:20])" 2>/dev/null)
        if [ -n "$R" ]; then
            echo "TP=$TP WINDOW=$WINDOW" > $SCRATCH/gate_verdict.txt
            echo "VERDICT: TP=$TP WINDOW=$WINDOW (reply: $R)"
            kill $PID 2>/dev/null
            exit 0
        fi
    fi
    echo "$WINDOW" >> $FAILED_LIST
    echo "window=$WINDOW failed"; tail -3 $LOG | cut -c1-140
    kill -9 $PID 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null
done
echo "VERDICT: NONE - even 65536 fails; stack-level problem"
exit 1
