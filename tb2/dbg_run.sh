#!/bin/bash
# Debug-partition validation of eval_tb2.sh: separate served name (own jobs
# dir), capped runtime; success = harbor gets past config and starts trials.
# debugjob PURGES modules and srun shells have no Lmod init: bootstrap it,
# and always launch with `srun -n1` (debugjob allocates ntasks=24).
source /etc/profile.d/*lmod*.sh 2>/dev/null || source /etc/profile
cd $SCRATCH/dccb
timeout 1700 bash tb2/eval_tb2.sh Qwen/Qwen3.5-35B-A3B qwen35-35b-dbg 1 easy25
echo "DEBUGRUN exit=$?"
