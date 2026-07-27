#!/bin/bash
# Where does vllm serve hang? Staged foreground diagnostic with timestamps.
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-0:45
#SBATCH --output=vllmdiag_%j.out
set -u
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
module load cuda/12.9 opencv python/3.12 2>/dev/null
source /scratch/anangia/ENV-vllm2/bin/activate
echo "[$(date +%T)] stage 1: bare import"
timeout 300 python -c "import time; t=time.time(); import vllm; print(f'import ok {time.time()-t:.0f}s', flush=True)" || echo "IMPORT HUNG/FAILED"
echo "[$(date +%T)] stage 2: engine init (LLM class, no server)"
timeout 600 python -c "
from vllm import LLM
print('constructing...', flush=True)
llm = LLM(model='Qwen/Qwen3.5-9B', max_model_len=8192, gpu_memory_utilization=0.9)
print('ENGINE UP', flush=True)
" || echo "ENGINE HUNG/FAILED"
echo "[$(date +%T)] stage 3: vllm serve foreground (2 min window)"
timeout 120 vllm serve Qwen/Qwen3.5-9B --port 18123 --max-model-len 8192 2>&1 | head -20
echo "[$(date +%T)] diag done"
