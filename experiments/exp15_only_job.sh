#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-8:00
#SBATCH --output=exp15b_%j.out
module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/does-compression-change-behavior/experiments
python exp15_real_boundaries.py
