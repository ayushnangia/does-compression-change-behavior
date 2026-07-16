#!/bin/bash
# The rate-distortion curve (exp6): 4 compressors x 4 budgets x 16 examples.
#   sbatch --account=def-rgrosse experiments/exp6_job.sh   (from repo root)
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-08:00
#SBATCH --output=exp6_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
python exp6_rate_distortion.py --examples-file ../examples_prefetched.json --num-examples 16
