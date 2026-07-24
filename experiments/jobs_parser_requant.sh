#!/bin/bash
# Quantify the native-format parser blind spot: rerun exp4 (the halting
# finding) with the FIXED parser. If halt rates reproduce (0.47 vs 0.19),
# finding 1 stands; the gap between old and new runs bounds the bug's bite.
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-8:00
#SBATCH --output=requant_%j.out
module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source /scratch/anangia/ENV-compress2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /scratch/anangia/dccb/experiments
python exp4_block_ablation.py --examples-file ../data/examples_64.json
