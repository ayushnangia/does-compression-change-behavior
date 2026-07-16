#!/bin/bash
# E-B: DPO-train the compressor on gap-filtered pairs, then held-out eval.
# Submit with dependency on exp11:
#   sbatch --account=def-rgrosse --dependency=afterok:<exp11-id> experiments/dpo_job.sh
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-10:00
#SBATCH --output=dpo_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
python dpo_train.py --pairs results/summary_pairs.jsonl
python evaluate_compressor.py --examples-file ../examples_64.json \
    --adapter results/dpo_compressor --train-cutoff 24 --num-eval 24
