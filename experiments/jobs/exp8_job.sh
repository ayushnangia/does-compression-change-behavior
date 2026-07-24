#!/bin/bash
# exp8: grounded agreement with the ORIGINAL agent's logged actions.
#   sbatch --account=def-rgrosse experiments/exp8_job.sh   (from repo root)
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00
#SBATCH --output=exp8_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
python exp8_grounded_agreement.py --examples-file ../data/examples_64.json --num-examples 32
