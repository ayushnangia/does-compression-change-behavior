#!/bin/bash
# exp14 arm 3: Qwen3.5-35B-A3B (bf16) — the exact MoE twin of GLM-4.7-Flash
# (both ~3B-active MoE) under the identical fragility matrix.
#   (auto-submitted by prep_35b.sh after the download completes)
#SBATCH --gpus-per-node=a100:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=0-3:00
#SBATCH --output=exp14_35b_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True

cd ~/does-compression-change-behavior/experiments

# tokenizer-identity guard: examples_64 is Qwen3.5-9B-tokenized
python - <<'EOF'
from transformers import AutoTokenizer
a = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
b = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)
s = "def f():\n  return 1  # <tool_calls>"
assert a(s)["input_ids"] == b(s)["input_ids"], "tokenizer mismatch - re-prefetch!"
print("35B-A3B tokenizer identical: examples_64 reusable")
EOF

python exp14_interface_fragility.py --model Qwen/Qwen3.5-35B-A3B \
    --examples-file ../examples_64.json
