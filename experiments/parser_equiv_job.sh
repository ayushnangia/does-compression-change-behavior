#!/bin/bash
# CPU-only: certify our parser mirror against vLLM's authority parsers.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-0:30
#SBATCH --output=parseq_%j.out
module load cuda/12.9 opencv python/3.12 2>/dev/null
source /scratch/anangia/ENV-vllm2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
cd /scratch/anangia/dccb/experiments
python parser_equivalence.py
