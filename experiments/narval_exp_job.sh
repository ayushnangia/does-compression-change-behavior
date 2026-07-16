#!/bin/bash
# Run the behavioral experiments (exp3-exp5) on 1x A100.
# Prereq (login node, once):
#   module load gcc arrow && source ~/ENV-compress/bin/activate
#   python prefetch.py --num-examples 24 --context-tokens 4096 --recent-tokens 512
#
#   sbatch experiments/narval_exp_job.sh          # from the repo root
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00
#SBATCH --output=exp_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate

export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
EX=../examples_prefetched.json

python exp3_target_stability.py  --examples-file $EX --num-examples 16
python exp4_block_ablation.py    --examples-file $EX --num-examples 16
python exp5_format_vs_content.py --examples-file $EX --num-examples 16
