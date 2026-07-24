#!/bin/bash
# Disambiguate the on-policy exp6 negative: label-level ceiling vs true
# no-effect, via the coarse tool-level metric.
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-8:00
#SBATCH --output=exp6c_%j.out
module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd ~/does-compression-change-behavior/experiments
python exp6_rate_distortion.py --examples-file ../data/examples_onpolicy.json --num-examples 14
