#!/bin/bash
# T4 resume: pairs already generated (177 written, 86 gap-filtered).
# Train + eval only, with the fragmentation fix.
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=0-8:00
#SBATCH --output=t4b_%j.out
module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source /scratch/anangia/ENV-compress2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # torch 2.9 reads the OLD name
cd /scratch/anangia/dccb/experiments
python dpo_train.py --pairs results/summary_pairs_16k.jsonl \
    --out results/dpo_compressor_r2 --epochs 3
python evaluate_compressor.py --examples-file ../examples_16k_large.json \
    --adapter results/dpo_compressor_r2 \
    --heldout-tasks results/heldout_tasks.json --num-eval 24 --summary-samples 2
