#!/bin/bash
# E-A: best-of-N selection by D (winner's-curse-guarded) + DPO pair generation.
#   sbatch --account=def-rgrosse experiments/exp11_job.sh
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-10:00
#SBATCH --output=exp11_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
python exp11_best_of_n.py --examples-file ../examples_64.json --num-examples 24
