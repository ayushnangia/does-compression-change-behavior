#!/bin/bash
# exp17 (minimal behavioral core, down to 2% kept) + long-context grounding
# arms at 32k and 64k (exp8/exp4 on window-filling on-policy examples).
#   sbatch --account=def-zhijing experiments/exp17_18_job.sh  (from $SCRATCH/dccb)
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --output=exp17_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source /scratch/anangia/ENV-compress2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/anangia/dccb/experiments

python exp17_minimal_core.py --examples-file ../data/examples_16k_large.json --num-examples 24
python exp8_grounded_agreement.py --examples-file ../data/examples_32k.json --num-examples 20
python exp4_block_ablation.py     --examples-file ../data/examples_32k.json --num-examples 16
# 64k: reduced samples to fit HF-generate KV on one A100 (linear-attention arch)
python exp8_grounded_agreement.py --examples-file ../data/examples_64k.json --num-examples 12 --samples 4
