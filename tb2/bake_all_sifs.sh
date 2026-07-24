#!/bin/bash
# Pre-bake all Terminal-Bench 2.0 task images into offline-ready .sif files.
#
# RUN ON THE LOGIN NODE (needs internet to pull from Docker Hub).
# Resumable: skips images whose .sif already exists. ~1-2 min per image.
#
#   nohup bash bake_all_sifs.sh > bake.log 2>&1 &
#
# Each image is baked with everything harbor's singularity bootstrap.sh would
# otherwise install at container start (python3, tmux, asciinema, and the
# /opt/harbor-server venv with uvicorn+fastapi) so task containers need ZERO
# internet at runtime — required on Narval compute nodes.
set -u

TB2=$SCRATCH/tb2
TEMPLATE=$TB2/defs/bake.def.tpl
CACHE=$TB2/sif_cache
mkdir -p "$CACHE" "$TB2/defs"

module load apptainer 2>/dev/null
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
export APPTAINER_TMPDIR=/tmp/$USER-apt-build   # tmpfs: dpkg on lustre is 30x slower
mkdir -p "$APPTAINER_TMPDIR"

ok=0; fail=0; skip=0
for toml in "$TB2"/terminal-bench/*/task.toml; do
    img=$(grep -oP 'docker_image = "\K[^"]+' "$toml")
    [ -z "$img" ] && continue
    safe=$(echo "$img" | tr '/:' '__')
    if [ -f "$CACHE/$safe.sif" ]; then
        skip=$((skip+1)); continue
    fi
    echo "[$(date +%H:%M:%S)] baking $img"
    sed "s|{{IMAGE}}|$img|" "$TEMPLATE" > "$TB2/defs/$safe.def"
    if apptainer build --force "$CACHE/$safe.sif" "$TB2/defs/$safe.def" \
            > "$TB2/defs/$safe.build.log" 2>&1; then
        ok=$((ok+1))
    elif grep -q "signal: killed" "$TB2/defs/$safe.build.log"; then
        # big image: squashfs OOM-killed on tmpfs -> retry on lustre (slower
        # but not memory-bound). apt is still fast: layers already cached.
        echo "  OOM on tmpfs — retrying with scratch tmpdir"
        mkdir -p "$SCRATCH/apptainer_tmp"
        if APPTAINER_TMPDIR=$SCRATCH/apptainer_tmp \
                apptainer build --force "$CACHE/$safe.sif" "$TB2/defs/$safe.def" \
                >> "$TB2/defs/$safe.build.log" 2>&1; then
            ok=$((ok+1))
        else
            fail=$((fail+1)); echo "  FAILED — see $TB2/defs/$safe.build.log"
            rm -f "$CACHE/$safe.sif"
        fi
    else
        fail=$((fail+1))
        echo "  FAILED — see $TB2/defs/$safe.build.log"
        rm -f "$CACHE/$safe.sif"
        # space out docker hub pulls: anonymous rate limit on shared IP
        sleep 20
    fi
done
echo "done: $ok built, $skip already cached, $fail failed"
