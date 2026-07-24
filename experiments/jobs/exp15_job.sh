#!/bin/bash
# exp15: score PRODUCTION compactions at the 23 real window-exhaustion
# boundaries + 16k pilot arms (exp4/exp8 on data/examples_16k, HF path).
#   sbatch --account=def-zhijing experiments/exp15_job.sh   (repo root)
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00
#SBATCH --output=exp15_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd ~/does-compression-change-behavior/experiments

python exp15_real_boundaries.py
python exp8_grounded_agreement.py --examples-file ../data/examples_16k.json --num-examples 24
python exp4_block_ablation.py     --examples-file ../data/examples_16k.json --num-examples 16
