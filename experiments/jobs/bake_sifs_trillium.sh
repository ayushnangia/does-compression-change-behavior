#!/bin/bash
# LOGIN NODE (Trillium): pre-bake TB2 task sifs into harbor's cache layout.
# Compute nodes have no internet, so every docker image referenced by the
# tasks in tb2_bf16_config.yaml must exist as
#   $SCRATCH/tb2/sif_cache/<image with / and : replaced by _>.sif
# (naming per harbor/environments/singularity/singularity.py _convert_docker_to_sif)
#   setsid nohup bash experiments/bake_sifs_trillium.sh > $SCRATCH/bake_sifs.log 2>&1 &
set -u
module load apptainer 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
mkdir -p "$APPTAINER_CACHEDIR" "$SCRATCH/tb2/sif_cache"

CONFIG=${1:-$SCRATCH/dccb/experiments/tb2_bf16_config.yaml}
TASKS=$(grep -oP 'terminal-bench/\K[\w-]+' "$CONFIG")

for t in $TASKS; do
    toml=$SCRATCH/tb2/terminal-bench/$t/task.toml
    img=$(grep -oP 'docker_image\s*=\s*"\K[^"]+' "$toml")
    [ -n "$img" ] || { echo "SKIP $t: no docker_image"; continue; }
    case "$img" in *:*) ;; *) img="$img:latest";; esac
    safe=$(echo "$img" | tr '/:' '__')
    sif=$SCRATCH/tb2/sif_cache/$safe.sif
    if [ -s "$sif" ]; then echo "CACHED $t -> $safe.sif"; continue; fi
    echo "[$(date +%H:%M)] PULL $t <- $img"
    apptainer pull "$sif" "docker://$img" || echo "FAILED $t ($img)"
done
echo "[$(date +%H:%M)] bake done: $(ls $SCRATCH/tb2/sif_cache/*.sif 2>/dev/null | wc -l) sifs"
