#!/bin/bash
# Gate #1 as a batch job (queued through maintenance; fires when nodes return):
# serve bf16 Qwen3.5-35B-A3B on ONE H100-80GB and get a completion out of it.
# On success, the dependent TB2 job (tb2_bf16_trillium_job.sh) starts.
#   sbatch --account=def-zhijing experiments/smoke_35b_sbatch.sh   (from $SCRATCH/dccb)
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=1:00:00
#SBATCH --output=smoke35b_%j.out
set -u
bash $SCRATCH/dccb/experiments/smoke_35b_trillium.sh
