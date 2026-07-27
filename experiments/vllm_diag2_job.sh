#!/bin/bash
# Name the hanging frame: SIGABRT makes faulthandler dump the stack.
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-0:20
#SBATCH --output=vllmdiag2_%j.out
set -u
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
module load cuda/12.9 opencv python/3.12 2>/dev/null
source /scratch/anangia/ENV-vllm2/bin/activate
python -V; which python
echo "[$(date +%T)] import with stack dump on timeout:"
timeout -s ABRT 240 python -c "import vllm; print('import ok')" 2>&1 | tail -40
echo "[$(date +%T)] done"
