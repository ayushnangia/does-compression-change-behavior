#!/bin/bash
# ON-POLICY replication of the core trio: the model being measured generated
# these traces itself (our TB2 trajectories) — closes the off-policy confound.
#   sbatch --account=def-zhijing experiments/onpolicy_job.sh   (repo root)
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-10:00
#SBATCH --output=onpolicy_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd ~/does-compression-change-behavior/experiments
EX=../data/examples_onpolicy.json

python exp8_grounded_agreement.py --examples-file $EX --num-examples 32
python exp4_block_ablation.py     --examples-file $EX --num-examples 24
python exp6_rate_distortion.py    --examples-file $EX --num-examples 16
