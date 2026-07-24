#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-2:00
#SBATCH --output=exp22smoke_%j.out
set -u
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1
module load cuda/12.9 opencv python/3.12 2>/dev/null
source /scratch/anangia/ENV-vllm2/bin/activate
vllm serve Qwen/Qwen3.5-9B --port 8000 --served-model-name qwen35-9b \
  --max-model-len 32768 --gpu-memory-utilization 0.90 > vllm_exp22_$SLURM_JOB_ID.log 2>&1 &
VPID=$!
deactivate
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:8000/health >/dev/null && { echo "vllm up ${i}0s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "vllm died"; exit 1; }
  sleep 10
done
module load apptainer gcc arrow 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache APPTAINER_TMPDIR=$SLURM_TMPDIR
export PYTHONPATH=/scratch/anangia/exp22wt:${PYTHONPATH:-}
source /scratch/anangia/ENV-harbor2/bin/activate
harbor run -c /scratch/anangia/exp22wt/exp22/smoke_config.yaml -y
S=$?
kill $VPID 2>/dev/null
exit $S
