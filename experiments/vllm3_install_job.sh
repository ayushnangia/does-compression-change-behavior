#!/bin/bash
# Finish ENV-vllm3 (vllm on the known-good 3.11 stack) on a compute node.
# --no-index: wheelhouse only (no internet on compute nodes; vllm 0.25 is in it).
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-3:00
#SBATCH --output=vllm3inst_%j.out
set -u
module load python/3.11 gcc arrow cuda/12.9 opencv 2>/dev/null
source /scratch/anangia/ENV-vllm3/bin/activate
python -V
pip install --no-index vllm 2>&1 | tail -5
echo "=== consumption test ==="
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
timeout 300 python -c "import time; t=time.time(); import vllm; print(f'import vllm OK in {time.time()-t:.0f}s, version', vllm.__version__)" || echo "IMPORT FAILED"
