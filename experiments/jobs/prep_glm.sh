#!/bin/bash
# LOGIN NODE: download GLM-4.7-Flash (~60GB), prefetch GLM-tokenized examples,
# then submit the 2-GPU family job automatically.
#   setsid nohup bash experiments/prep_glm.sh > glm_prep.log 2>&1 &
set -e
module load python/3.11 gcc arrow 2>/dev/null
source ~/ENV-compress/bin/activate
export HF_HOME=$SCRATCH/hf
cd ~/does-compression-change-behavior

echo "[$(date +%H:%M)] downloading GLM-4.7-Flash ..."
hf download zai-org/GLM-4.7-Flash
echo "[$(date +%H:%M)] prefetching GLM-tokenized examples ..."
python prefetch.py --model zai-org/GLM-4.7-Flash --num-examples 32 \
    --context-tokens 4096 --recent-tokens 512 --out data/examples_glm.json
echo "[$(date +%H:%M)] submitting family job ..."
sbatch --account=def-rgrosse experiments/glm_family_job.sh
echo "[$(date +%H:%M)] done"
