#!/bin/bash
# exp14: interface-fragility matrix — GLM-30B-A3B vs Qwen3.5-27B (both bf16)
# under identical {wrapper vs native template} x {top_p 1.0 vs 0.9}.
#   sbatch --account=def-zhijing experiments/exp14_job.sh   (repo root)
#SBATCH --gpus-per-node=a100:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=0-3:00
#SBATCH --output=exp14_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True

cd ~/does-compression-change-behavior/experiments

python exp14_interface_fragility.py --model zai-org/GLM-4.7-Flash \
    --examples-file ../data/examples_glm.json
python exp14_interface_fragility.py --model Qwen/Qwen3.5-27B \
    --examples-file ../data/examples_64.json
