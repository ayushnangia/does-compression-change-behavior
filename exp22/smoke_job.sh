#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-2:00
#SBATCH --output=exp22smoke_%j.out
set -u
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1
PORT=$((10000 + SLURM_JOB_ID % 20000))   # unique per job: nodes are SHARED, :8000 collides
module load python/3.11 gcc arrow cuda/12.9 opencv 2>/dev/null
source /scratch/anangia/ENV-vllm3/bin/activate
vllm serve Qwen/Qwen3.5-9B --port $PORT --served-model-name qwen35-9b \
  --max-model-len 32768 --gpu-memory-utilization 0.90 > vllm_exp22_$SLURM_JOB_ID.log 2>&1 &
VPID=$!
deactivate
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:$PORT/health >/dev/null && { echo "vllm up ${i}0s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "vllm died"; exit 1; }
  [ $((i % 12)) -eq 0 ] && echo "waiting for vllm... ${i}0s (log: $(wc -c < vllm_exp22_$SLURM_JOB_ID.log) bytes)"
  sleep 10
done
# HARD GUARD (dropped in the first version - attempts 3 and 5 ran harbor
# against a dead port because the loop fell through silently):
if ! curl -s http://127.0.0.1:$PORT/health >/dev/null; then
  echo "vllm never came up; last log lines:"; tail -20 vllm_exp22_$SLURM_JOB_ID.log
  kill $VPID 2>/dev/null; exit 1
fi
module load apptainer gcc arrow 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache APPTAINER_TMPDIR=$SLURM_TMPDIR
export PYTHONPATH=/scratch/anangia/exp22wt:${PYTHONPATH:-}
source /scratch/anangia/ENV-harbor2/bin/activate
CFG=$SLURM_TMPDIR/smoke_config.yaml
sed -e "s|http://127.0.0.1:8000|http://127.0.0.1:$PORT|" -e "s|job_name: exp22-smoke|job_name: exp22-smoke-$SLURM_JOB_ID|" /scratch/anangia/exp22wt/exp22/smoke_config.yaml > $CFG
harbor run -c $CFG -y
S=$?
kill $VPID 2>/dev/null
exit $S
