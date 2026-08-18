#!/bin/bash
# Prove that a rebaked TB2 image can score a known-good solution completely
# offline. This is a hard gate: never quote Pass@1 unless this writes reward=1.
#
# Usage: bash tb2/oracle_smoke.sh [task]
# Default task is in easy25 and exercises uvx + Selenium.
set -euo pipefail

TASK=${1:-break-filter-js-from-html}
ROOT=${TB2_DIR:-$SCRATCH/tb2}
TASK_DIR=$ROOT/terminal-bench/$TASK
TOML=$TASK_DIR/task.toml
[ -f "$TOML" ] || { echo "missing task: $TASK_DIR" >&2; exit 2; }
[ -f "$TASK_DIR/solution/solve.sh" ] || { echo "missing oracle solve.sh" >&2; exit 2; }

IMG=$(grep -oP 'docker_image = "\K[^"]+' "$TOML")
SAFE=$(echo "$IMG" | tr '/:' '__')
SIF=$ROOT/sif_cache/$SAFE.sif
[ -f "$SIF" ] || { echo "missing SIF: $SIF" >&2; exit 2; }

module load apptainer 2>/dev/null || true
LOGS=$(mktemp -d "$SCRATCH/oracle-${TASK}.XXXXXX")
mkdir -p "$LOGS/verifier"

echo "task=$TASK image=$IMG logs=$LOGS"
# Proxy only proves external networking is unavailable. localhost MUST bypass
# it: browser-based verifiers use a local Selenium driver.
set +e
timeout 900 apptainer exec --fakeroot --writable-tmpfs \
  --bind "$TASK_DIR/tests:/tests,$TASK_DIR/solution:/solution:ro,$LOGS:/logs" \
  "$SIF" bash -lc '
    export HOME=/tmp/oracle-home UV_OFFLINE=1 PATH=/usr/local/bin:$PATH
    export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9
    export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
    mkdir -p "$HOME"; cd /app
    bash /solution/solve.sh
    bash /tests/test.sh
  ' 2>&1 | tee "$LOGS/oracle.out"
RC=${PIPESTATUS[0]}
set -e
REWARD=$(cat "$LOGS/verifier/reward.txt" 2>/dev/null || true)
if [ "$RC" -ne 0 ] || [ "$REWARD" != "1" ]; then
  echo "ORACLE GATE FAILED: rc=$RC reward=${REWARD:-missing}" >&2
  exit 1
fi
echo "ORACLE GATE GREEN: reward=1"
