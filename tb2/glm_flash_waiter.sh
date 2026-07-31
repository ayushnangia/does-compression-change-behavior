#!/bin/bash
# LOGIN NODE: wait for zai-org/GLM-4.7-Flash to finish downloading, verify all
# shards, then submit the smoke -> TB2-easy25 chain under def-rgrosse.
#   setsid nohup bash tb2/glm_flash_waiter.sh > $SCRATCH/glm_waiter.log 2>&1 &
set -u
MODEL_DIR=$SCRATCH/hf/hub/models--zai-org--GLM-4.7-Flash
cd $SCRATCH/dccb

for i in $(seq 1 240); do   # up to 4h
    if ! pgrep -f "hf download zai-org/GLM-4.7-Flash" >/dev/null; then
        snap=$(ls -d $MODEL_DIR/snapshots/*/ 2>/dev/null | head -1)
        if [ -n "$snap" ] && [ -f "$snap/config.json" ] && \
           [ "$(find $MODEL_DIR -name '*.incomplete' | wc -l)" = 0 ]; then
            echo "[$(date +%H:%M)] download complete: $snap"
            break
        fi
        echo "[$(date +%H:%M)] downloader gone but snapshot incomplete - resuming"
        module load gcc python/3.11 arrow/19.0.1 2>/dev/null
        source $HOME/ENV-compress2/bin/activate
        HF_HOME=$SCRATCH/hf hf download zai-org/GLM-4.7-Flash >> $SCRATCH/prefetch_glm.log 2>&1 &
        deactivate
    fi
    sleep 60
done

SMOKE=$(SBATCH_ACCOUNT=def-rgrosse sbatch --parsable --gpus-per-node=h100:1 --time=1:00:00 \
    tb2/smoke_serve.sh zai-org/GLM-4.7-Flash glm47-flash) || exit 1
TB2=$(SBATCH_ACCOUNT=def-rgrosse sbatch --parsable --dependency=afterok:$SMOKE \
    --gpus-per-node=h100:1 --time=0-16:00 \
    tb2/eval_tb2.sh zai-org/GLM-4.7-Flash glm47-flash 1 easy25) || exit 1
echo "[$(date +%H:%M)] chain submitted: smoke=$SMOKE -> tb2=$TB2"
