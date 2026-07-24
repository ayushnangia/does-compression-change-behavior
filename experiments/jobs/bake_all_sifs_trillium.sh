#!/bin/bash
# LOGIN NODE (Trillium): bake sifs for ALL downloaded TB2 tasks (not just the
# easy-25), so any future config is turnkey. Same naming as harbor's cache.
#   setsid nohup bash experiments/bake_all_sifs_trillium.sh > $SCRATCH/bake_all_sifs.log 2>&1 &
set -u
module load apptainer 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
mkdir -p "$APPTAINER_CACHEDIR" "$SCRATCH/tb2/sif_cache"

for toml in $SCRATCH/tb2/terminal-bench/*/task.toml; do
    t=$(basename $(dirname "$toml"))
    img=$(grep -oP 'docker_image\s*=\s*"\K[^"]+' "$toml")
    [ -n "$img" ] || { echo "SKIP $t: no docker_image"; continue; }
    case "$img" in *:*) ;; *) img="$img:latest";; esac
    safe=$(echo "$img" | tr '/:' '__')
    sif=$SCRATCH/tb2/sif_cache/$safe.sif
    if [ -s "$sif" ]; then echo "CACHED $t"; continue; fi
    echo "[$(date +%H:%M)] PULL $t <- $img"
    apptainer pull "$sif" "docker://$img" || echo "FAILED $t ($img)"
done
echo "[$(date +%H:%M)] bake done: $(ls $SCRATCH/tb2/sif_cache/*.sif 2>/dev/null | wc -l) sifs"
