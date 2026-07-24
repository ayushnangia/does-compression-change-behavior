#!/bin/bash
# TB2 easy-25 with bf16 Qwen3.5-35B-A3B (TP=4), 32k window, 4x task timeouts.
#   sbatch --account=def-zhijing experiments/tb2_bf16_job.sh  (from $SCRATCH/dccb)
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=0-16:00
#SBATCH --output=tb2bf16_%j.out
set -u

export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1

module load cuda/12.9 opencv python/3.12 2>/dev/null
source /scratch/anangia/ENV-vllm2/bin/activate
stdbuf -oL -eL vllm serve Qwen/Qwen3.5-35B-A3B --port 8000 \
    --served-model-name qwen35-35b-bf16 \
    --tensor-parallel-size 4 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 > vllm_bf16_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
deactivate

for i in $(seq 1 270); do
    curl -s http://127.0.0.1:8000/health > /dev/null && { echo "vLLM up after ${i}0s"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; exit 1; }
    sleep 10
done
curl -s http://127.0.0.1:8000/health > /dev/null || { echo "vLLM never came up"; exit 1; }

module load apptainer gcc arrow 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache APPTAINER_TMPDIR=$SLURM_TMPDIR
source /scratch/anangia/ENV-harbor2/bin/activate

harbor run -c /scratch/anangia/dccb/experiments/tb2_bf16_config.yaml \
    --agent-timeout-multiplier 4 -y
STATUS=$?
kill $VLLM_PID 2>/dev/null
exit $STATUS
