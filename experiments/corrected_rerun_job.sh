#!/bin/bash
# CORRECTED reruns after the soundness audit:
#  - exp4: role-aware block regexes (v1 'drop_observations' was a no-op)
#  - exp3: + padding control (separates instability from information loss)
#  - exp6: properly-labeled hierarchical block_aware + wrapper budget fix
#  - exp12: portability matrix (was missing entirely)
#   sbatch --account=def-rgrosse experiments/corrected_rerun_job.sh
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00
#SBATCH --output=corrected_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
EX=../examples_64.json

python exp4_block_ablation.py   --examples-file $EX --num-examples 24
python exp3_target_stability.py --examples-file $EX --num-examples 16
python exp6_rate_distortion.py  --examples-file $EX --num-examples 16
python exp12_portability.py     --examples-file $EX --num-examples 16
