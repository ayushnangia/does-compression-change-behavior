#!/bin/bash
# LOGIN NODE, detached: makes the gate->relaunch stage self-healing.
# The orchestrator can die without submitting TB2 jobs in exactly one way:
# the debugjob gate times out in queue (6500s incl. wait) or hits the
# wrapper's cancel-prompt. This supervisor retries the orchestration up to
# 3 total attempts; success = $SCRATCH/tb2_jobs.txt exists.
#   setsid nohup bash tb2/chain_supervisor.sh > $SCRATCH/chain_supervisor.log 2>&1 &
set -u
cd $SCRATCH/dccb
for attempt in 1 2 3; do
    # wait for any running orchestrator to finish
    while pgrep -f "bash tb2/relaunch_orchestrator.sh" >/dev/null; do sleep 120; done
    if [ -s $SCRATCH/tb2_jobs.txt ]; then
        echo "[$(date +%H:%M)] chain healthy: TB2 jobs $(cat $SCRATCH/tb2_jobs.txt)"
        exit 0
    fi
    [ $attempt -eq 3 ] && break
    echo "[$(date +%H:%M)] orchestrator ended without submitting - retry $((attempt+1))"
    # clear any stale debugjob so the wrapper prompt cannot trigger
    squeue -u $USER -h -o "%j %A" | awk '/^debugjob-/{print $2}' | xargs -r scancel
    sleep 30
    SKIP_SIF_SWEEP=1 nohup bash tb2/relaunch_orchestrator.sh >> $SCRATCH/orchestrator.log 2>&1 < /dev/null
done
echo "[$(date +%H:%M)] FATAL after 3 attempts - human needed (gate cannot pass or cluster unhealthy)"
