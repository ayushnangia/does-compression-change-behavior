#!/bin/bash
# exp7 (compaction-chain compounding, 9B) + exp3 rerun at 4B (does target
# stability depend on model scale? — the thread's oracle-choice question).
#   sbatch --account=def-rgrosse experiments/exp7_job.sh   (from repo root)
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00
#SBATCH --output=exp7_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
EX=../data/examples_prefetched.json

# guard: prefetched examples are 9B-tokenized; only reuse them for the 4B if
# the tokenizers are actually identical (Qwen family usually shares vocab).
python - <<'EOF'
from transformers import AutoTokenizer
a = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
b = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
s = "def f(x):\n    return x  # <tool_calls> test 123"
assert a(s)["input_ids"] == b(s)["input_ids"], "tokenizers differ - re-prefetch for 4B!"
print("tokenizers identical: examples file reusable for 4B")
EOF

python exp7_compaction_chain.py --examples-file $EX --num-examples 16
python exp3_target_stability.py --model Qwen/Qwen3.5-4B --examples-file $EX --num-examples 16
