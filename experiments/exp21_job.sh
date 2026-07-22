#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --output=exp21_%j.out
module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source /scratch/anangia/ENV-compress2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /scratch/anangia/dccb/experiments
python exp21_canonical_skeleton.py --examples-file ../examples_16k_large.json --num-examples 24
