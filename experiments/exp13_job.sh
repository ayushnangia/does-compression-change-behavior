#!/bin/bash
# exp13: recoverable compression — deletion manifest vs plain summary,
# in both plain and recovery-scaffold modes.
#   sbatch --account=def-rgrosse experiments/exp13_job.sh   (from repo root)
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-10:00
#SBATCH --output=exp13_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd ~/does-compression-change-behavior/experiments
python exp13_manifest.py --examples-file ../examples_64.json \
    --num-examples 24 --scaffold-examples 12
