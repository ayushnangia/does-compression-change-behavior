#!/bin/bash
# T1 acceptance: HF-vs-vLLM scorer equivalence + throughput, one GPU slot.
#   sbatch --account=def-zhijing experiments/scorer_equiv_job.sh  (repo root)
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-4:00
#SBATCH --output=scorer_eq_%j.out

export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
cd ~/does-compression-change-behavior/experiments

echo "=== pass 1: HF reference (ENV-compress) ==="
module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
python scorer_equivalence.py --backend hf --out /tmp/eq_hf.json
deactivate; module purge 2>/dev/null

echo "=== pass 2: vLLM candidate (ENV-vllm) ==="
module load cuda/12.9 opencv python/3.12 2>/dev/null
source ~/ENV-vllm/bin/activate
python scorer_equivalence.py --backend vllm --out /tmp/eq_vllm.json
echo "=== compare ==="
python scorer_equivalence.py --compare /tmp/eq_hf.json /tmp/eq_vllm.json
