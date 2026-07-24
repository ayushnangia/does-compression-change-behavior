#!/bin/bash
# Statistical scale-up: headline experiments at N=48 with a second seed.
#   sbatch --account=def-rgrosse experiments/scale_job.sh   (from repo root)
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00
#SBATCH --output=scale_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd ~/does-compression-change-behavior/experiments
EX=../data/examples_64.json

python exp4_block_ablation.py   --examples-file $EX --num-examples 48 --seed 0
python exp4_block_ablation.py   --examples-file $EX --num-examples 48 --seed 1
python exp3_target_stability.py --examples-file $EX --num-examples 48 --seed 0
python exp6_rate_distortion.py  --examples-file $EX --num-examples 32 --seed 0
