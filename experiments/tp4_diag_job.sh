#!/bin/bash
# Isolate the TP=4 startup failure: engine init only, staged prints, no harbor.
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=0-1:30
#SBATCH --output=tp4diag_%j.out
module load cuda/12.9 opencv python/3.12 2>/dev/null
source /scratch/anangia/ENV-vllm2/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=DEBUG NCCL_DEBUG=WARN
nvidia-smi -L
echo "=== stage 1: TP=2 init (35B bf16, tight but diagnostic) ==="
timeout 1500 python -u -c "
from vllm import LLM
print('constructing TP=2...', flush=True)
llm = LLM(model='Qwen/Qwen3.5-35B-A3B', tensor_parallel_size=2,
          max_model_len=8192, gpu_memory_utilization=0.95)
print('TP=2 UP', flush=True)
out = llm.generate(['Hello'],)
print('TP=2 GENERATES:', out[0].outputs[0].text[:50], flush=True)
" 2>&1 | tail -5
echo "=== stage 2: TP=4 init ==="
timeout 2400 python -u -c "
from vllm import LLM
print('constructing TP=4...', flush=True)
llm = LLM(model='Qwen/Qwen3.5-35B-A3B', tensor_parallel_size=4,
          max_model_len=32768, gpu_memory_utilization=0.92)
print('TP=4 UP', flush=True)
out = llm.generate(['Hello'],)
print('TP=4 GENERATES:', out[0].outputs[0].text[:50], flush=True)
" 2>&1 | tail -8
echo "diag done"
