#!/bin/bash
# Second model family: exp4 + exp9-style check with GLM-4.7-Flash (30B-A3B,
# bf16 ~60GB -> 2x A100 via device_map=auto). Rules out Qwen-specific effects.
# Prereq (login node): bash experiments/prep_glm.sh  (download + prefetch)
#   sbatch --account=def-rgrosse experiments/glm_family_job.sh  (from repo root)
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G

#SBATCH --time=0-10:00
#SBATCH --output=glm_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/does-compression-change-behavior/experiments
EX=../examples_glm.json

python exp4_block_ablation.py --model zai-org/GLM-4.7-Flash --examples-file $EX --num-examples 24
python exp7_compaction_chain.py --model zai-org/GLM-4.7-Flash --examples-file $EX --num-examples 16
