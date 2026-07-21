#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-3:00
#SBATCH --output=exp19_%j.out
module load cuda/12.9 opencv python/3.12 2>/dev/null
source /scratch/anangia/ENV-vllm2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
cd /scratch/anangia/dccb/experiments
python exp19_exact_distribution.py --examples-file ../examples_16k_large.json --num-examples 16
