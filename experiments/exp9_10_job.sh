#!/bin/bash
# exp9 (summary-policy audit/search) + exp10 (damage propagation).
#   sbatch --account=def-rgrosse experiments/exp9_10_job.sh   (from repo root)
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-10:00
#SBATCH --output=exp9_10_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
EX=../examples_64.json

python exp9_summary_policies.py --examples-file $EX --num-examples 24
python exp10_propagation.py     --examples-file $EX --num-examples 24
