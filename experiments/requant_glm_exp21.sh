#!/bin/bash
# Requant the two most at-risk numbers under the fixed parser + deployment
# budget: exp14 GLM arm (native arg_key format was parsed as halt - the
# 0.00 wrapper collapse may be partly parser blindness) and exp21
# (wrapped-vs-bare +19pts - native-format fallback in the bare condition
# would inflate it).
#SBATCH --gpus-per-node=a100:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=0-12:00
#SBATCH --output=requant2_%j.out
module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source /scratch/anangia/ENV-compress2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /scratch/anangia/dccb/experiments
echo "=== exp14 GLM requant (fixed parser) ==="
python exp14_interface_fragility.py --model zai-org/GLM-4.7-Flash \
    --examples-file ../data/examples_glm.json --num-examples 6 --samples 4
echo "=== exp21 requant (fixed parser, deployment budget) ==="
python exp21_canonical_skeleton.py --examples-file ../data/examples_16k_large.json --num-examples 24
