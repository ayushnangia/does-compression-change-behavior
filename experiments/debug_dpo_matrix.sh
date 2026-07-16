#!/bin/bash
# DPO debug matrix v2: persistent logs, full errors, with/without fla kernels.
#   sbatch --account=def-zhijing experiments/debug_dpo_matrix.sh
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-2:00
#SBATCH --output=dpo_matrix_%j.out

module load cuda/12.9 python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd ~/does-compression-change-behavior/experiments
DBG=$SCRATCH/dpo_dbg; mkdir -p $DBG
# use ALL pairs with min-gap 0: 19 pairs clears the >=8 reliability guard
# (v2's 4-pair subset tripped the production safety rail — tested nothing)
cp results/summary_pairs.jsonl $DBG/pairs.jsonl

run_cfg () {  # name, visible devices, extra args
    echo "=== CONFIG: $1 (gpus=$2) ==="
    CUDA_VISIBLE_DEVICES=$2 timeout 1200 python dpo_train.py \
        --pairs $DBG/pairs.jsonl --out $DBG/$1 --epochs 1 --min-gap 0.0 $3 \
        > $DBG/$1.log 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f $DBG/$1/adapter_config.json ]; then
        echo "PASS: $1"
    else
        echo "FAIL: $1 (rc=$rc) --- error tail:"
        grep -vE "Loading weights|it/s\]|s/it\]" $DBG/$1.log | tail -12
    fi
}

echo "########## WITH fla kernels ##########"
python -c "import fla; print('fla', fla.__version__)" 2>/dev/null || echo "fla absent"
run_cfg A_2gpu_fla "0,1" ""
run_cfg B_1gpu_fla "0" ""

echo "########## WITHOUT fla kernels (uninstall is offline-safe) ##########"
pip uninstall -y -q flash-linear-attention fla-core 2>/dev/null
python -c "import fla" 2>/dev/null && echo "WARN: fla still importable" || echo "fla removed"
run_cfg C_2gpu_nofla "0,1" ""
run_cfg D_1gpu_nofla "0" ""
run_cfg E_2gpu_nofla_nockpt "0,1" "--no-grad-ckpt"
echo "matrix v2 done"
