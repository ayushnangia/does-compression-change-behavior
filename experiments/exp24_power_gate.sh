#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=exp24powergate
#SBATCH --time=0-00:10
#SBATCH --output=exp24_power_gate_%j.out
# CPU-only integrity/power gate. Full-node GRPO jobs depend on afterok here.
set -euo pipefail
DATA_DIR=${1:-experiments/results/exp24_qwen38_data}
MIN_TRAIN=${2:-1000}
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}
[ -f "$ROOT/experiments/exp24_grpo_train.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
module load StdEnv/2023
module load gcc python/3.11 python-build-bundle/2026a
REAL_HOME=$HOME
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
$REAL_HOME/ENV-compress2/bin/python - "$DATA_DIR" "$MIN_TRAIN" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

data = Path(sys.argv[1])
minimum = int(sys.argv[2])
manifest_path = data / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"POWER GATE REFUSED: missing {manifest_path}")
manifest = json.loads(manifest_path.read_text())
if manifest.get("source_model") != "Qwen/Qwen3.8-27B":
    raise SystemExit(f"POWER GATE REFUSED: lineage={manifest.get('source_model')!r}")

seen_tasks = {}
counts = {}
for split in ("train", "validation", "test"):
    path = data / f"{split}.jsonl"
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"POWER GATE REFUSED: empty {split}")
    tasks = set()
    n = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            n += 1
            if row.get("source_model") != "Qwen/Qwen3.8-27B":
                raise SystemExit(f"POWER GATE REFUSED: off-policy row in {split}")
            prompt = row.get("prompt")
            if not (isinstance(prompt, list) and prompt and
                    prompt[0].get("role") == "user"):
                raise SystemExit(f"POWER GATE REFUSED: non-chat prompt in {split}")
            tasks.add(row["task"])
    expected = manifest["splits"][split]
    if n != expected["examples"] or tasks != set(expected["tasks"]):
        raise SystemExit(f"POWER GATE REFUSED: manifest mismatch in {split}")
    counts[split] = n
    seen_tasks[split] = tasks

for a, b in (("train", "validation"), ("train", "test"),
             ("validation", "test")):
    overlap = seen_tasks[a] & seen_tasks[b]
    if overlap:
        raise SystemExit(f"POWER GATE REFUSED: {a}/{b} overlap={sorted(overlap)}")
if counts["train"] < minimum:
    raise SystemExit(f"POWER GATE REFUSED: train={counts['train']} < {minimum}")

source = Path(manifest["source_file"])
if source.is_file():
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != manifest["source_sha256"]:
        raise SystemExit("POWER GATE REFUSED: source SHA mismatch")
print("EXP24 POWER GATE GREEN", json.dumps(counts, sort_keys=True),
      "task-disjoint=true lineage=Qwen/Qwen3.8-27B")
PY
