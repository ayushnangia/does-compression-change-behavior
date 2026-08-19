#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=q38data
#SBATCH --time=0-06:00
#SBATCH --output=exp24_data_%j.out
# Submit with --dependency=afterok:<qwen38 baseline job>.
# Optional args: <trajectory-glob> <example-json> <prepared-data-dir>
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}; [ -f "$ROOT/behavior.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
module load gcc cuda python/3.11 arrow/19.0.1 2>/dev/null
REAL_HOME=$HOME; export HOME=$SCRATCH/compute_home HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$HOME/.cache"
GLOB=${1:-"$SCRATCH/tb2/jobs/tb2-qwen38-27b-valid-easy25/*/agent/trajectory.json"}
EXAMPLES=${2:-data/examples_qwen38_exp24.json}
OUT_DIR=${3:-experiments/results/exp24_qwen38_data}
$REAL_HOME/ENV-compress2/bin/python experiments/exp24_prefetch_onpolicy.py \
  --model Qwen/Qwen3.8-27B --traj-glob "$GLOB" \
  --context-tokens 65536 --min-context-tokens 8192 --recent-tokens 4096 \
  --num-examples 2000 --max-per-task 100 \
  --out "$EXAMPLES"
$REAL_HOME/ENV-compress2/bin/python experiments/exp24_prepare.py \
  --model Qwen/Qwen3.8-27B --examples-file "$EXAMPLES" \
  --out-dir "$OUT_DIR"
