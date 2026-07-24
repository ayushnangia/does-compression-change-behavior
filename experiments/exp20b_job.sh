#!/bin/bash
# exp20b: the containment law under CONTINUOUS rate variation (hostile-review
# fix for "correlation driven by 6 condition clusters"). Three rates x N=48
# on 16k on-policy data; pooled Spearman across rates gives within-condition
# rate-driven variation. Runs under the fixed parser + deployment budget.
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --output=exp20b_%j.out
module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source /scratch/anangia/ENV-compress2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /scratch/anangia/dccb/experiments
for R in 0.05 0.125 0.5; do
  echo "=== rate $R ==="
  python exp20_ood_bridge.py --examples-file ../data/examples_16k_large.json \
      --num-examples 48 --rate $R --seed 1
done
