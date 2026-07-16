#!/bin/bash
# 45-min GLM parser check (verdict on the quarantined exp7 numbers).
#   sbatch --account=def-zhijing experiments/glm_check_job.sh   (repo root)
#SBATCH --gpus-per-node=a100:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=0-0:45
#SBATCH --output=glm_check_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/does-compression-change-behavior/experiments
python debug_glm_summary.py --examples-file ../examples_glm.json
