#!/bin/bash
# Full behavior-measurement run on 1x A100-40GB (Narval).
#   sbatch narval_job.sh
# Prereq (once, on the LOGIN node — compute nodes have no internet):
#   python prefetch.py --model Qwen/Qwen3.5-9B --num-examples 32
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-08:00
#SBATCH --output=%N-%j.out

module load cuda/12.9 python/3.11 gcc arrow
source ~/ENV-compress/bin/activate

export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior

# pass 1: plain next-action comparison
python run_experiment.py \
  --model Qwen/Qwen3.5-9B \
  --examples-file examples_prefetched.json \
  --num-examples 32 --samples 8

# pass 2: the recovery test (tool menu + lookups served from real history)
python run_experiment.py \
  --model Qwen/Qwen3.5-9B \
  --examples-file examples_prefetched.json \
  --num-examples 32 --samples 8 --scaffold
