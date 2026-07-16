#!/bin/bash
# T4 round 2: on-policy pairs at 16k (vLLM, batched) -> DPO -> powered
# task-split eval. Every round-1 root cause fixed.
#   sbatch --account=def-zhijing experiments/t4_round2_job.sh   (repo root)
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=0-12:00
#SBATCH --output=t4_%j.out

export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
cd ~/does-compression-change-behavior/experiments

echo "===== phase 1: pair generation (ENV-vllm, 1 GPU) ====="
module load cuda/12.9 opencv python/3.12 2>/dev/null
source ~/ENV-vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python exp16_pairs_at_scale.py \
    --examples-file ../examples_16k_large.json --num-examples 280
deactivate; module purge 2>/dev/null

echo "===== phase 2: DPO round 2 (ENV-compress, 2 GPUs) ====="
module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
python dpo_train.py --pairs results/summary_pairs_16k.jsonl \
    --out results/dpo_compressor_r2 --epochs 3

echo "===== phase 3: powered task-split eval ====="
python evaluate_compressor.py --examples-file ../examples_16k_large.json \
    --adapter results/dpo_compressor_r2 \
    --heldout-tasks results/heldout_tasks.json --num-eval 24 \
    --summary-samples 2
