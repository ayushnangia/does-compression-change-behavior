#!/bin/bash
# 30-min diagnostic: what does the model actually generate at decision points?
#   sbatch --account=def-rgrosse experiments/debug_job.sh    (from repo root)
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=debug_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
python debug_continuations.py --examples-file ../examples_prefetched.json
