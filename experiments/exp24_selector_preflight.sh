#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=exp24selgate
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-01:00
#SBATCH --output=exp24_selector_preflight_%j.out
# One-GPU compute-node gate: load Qwen3.5-4B, attach LoRA, and construct the
# installed TRL GRPOTrainer before allocating a four-H100 executor node.
set -euo pipefail
DATA_DIR=${1:-experiments/results/exp24_qwen38_pilot_data}
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}
[ -f "$ROOT/experiments/exp24_grpo_train.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
# Loading StdEnv changes MODULEPATH, so this must be a separate transaction
# on Trillium's pristine --export=NONE compute shell.
module load StdEnv/2023
module load gcc cuda python/3.11 python-build-bundle/2026a \
  scipy-stack/2026a arrow/19.0.1
module list
REAL_HOME=$HOME
export HOME=$SCRATCH/compute_home HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
# Alliance's existing PYTHONPATH contains sitecustomize, which exposes the
# packages in EBPYTHONPREFIXES. Prepend the repo; do not overwrite that path.
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
mkdir -p "$HOME/.cache" "$SLURM_TMPDIR/triton-selector-preflight"
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$SLURM_TMPDIR/triton-selector-preflight \
  $REAL_HOME/ENV-compress2/bin/python experiments/exp24_grpo_train.py \
  --train-file "$DATA_DIR/train.jsonl" --out "$SLURM_TMPDIR/unused" \
  --reward-log "$SLURM_TMPDIR/unused-rewards.jsonl" \
  --max-steps 1 --min-train-examples 20 --preflight-only
