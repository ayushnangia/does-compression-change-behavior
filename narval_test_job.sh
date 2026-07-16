#!/bin/bash
# 20-minute smoke test on 1x A100 — ALWAYS run this before the real job.
#   sbatch narval_test_job.sh
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0:20:00
#SBATCH --output=%N-%j.out

module load python/3.11 gcc arrow
source ~/ENV-compress/bin/activate

export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior
python run_experiment.py \
  --model Qwen/Qwen3.5-9B \
  --examples-file examples_prefetched.json \
  --num-examples 2 --samples 2
