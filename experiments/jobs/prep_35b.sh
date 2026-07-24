#!/bin/bash
# LOGIN NODE: download Qwen3.5-35B-A3B bf16 (~70GB), then auto-submit its
# exp14 fragility arm.
#   setsid nohup bash experiments/prep_35b.sh > prep_35b.log 2>&1 &
set -e
module load python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf
cd ~/does-compression-change-behavior

echo "[$(date +%H:%M)] downloading Qwen3.5-35B-A3B (bf16, ~70GB) ..."
hf download Qwen/Qwen3.5-35B-A3B
echo "[$(date +%H:%M)] submitting exp14 arm ..."
sbatch --account=def-zhijing experiments/exp14_35b_job.sh
echo "[$(date +%H:%M)] done"
