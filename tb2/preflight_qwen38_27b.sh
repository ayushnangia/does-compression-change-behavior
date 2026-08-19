#!/bin/bash
#SBATCH --account=def-rgrosse
#SBATCH --job-name=q38livegate
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-02:00
#SBATCH --output=q38_livegate_%j.out
# Full-stack gate for Qwen3.8-27B. Do not approximate Harbor with a raw
# completions prompt: run one real Terminus-2 task, then require Harbor-parsed
# commands AND a verifier that actually executed.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$SCRATCH/dccb}; [ -f "$ROOT/behavior.py" ] || ROOT=$SCRATCH/dccb
cd "$ROOT"
SERVED="qwen38-27b-smoke-$SLURM_JOB_ID"
SUBSET="$ROOT/tb2/q38_smoke.txt"

# This starts vLLM at the production 77,824-token window, runs Harbor/Terminus,
# executes the rebaked offline verifier, and shuts the server down.
bash tb2/eval_tb2.sh Qwen/Qwen3.8-27B "$SERVED" 1 "$SUBSET" 4

JOB="$SCRATCH/tb2/jobs/tb2-$SERVED-q38_smoke"
TRIAL=$(find "$JOB" -mindepth 1 -maxdepth 1 -type d | head -1)
[ -n "$TRIAL" ] || { echo "LIVE GATE FAILED: no trial directory in $JOB"; exit 1; }
TRAJ="$TRIAL/agent/trajectory.json"
REWARD="$TRIAL/verifier/reward.txt"
TESTOUT="$TRIAL/verifier/test-stdout.txt"
[ -s "$TRAJ" ] || { echo "LIVE GATE FAILED: missing trajectory"; exit 1; }
[ ! -s "$TRIAL/exception.txt" ] || { echo "LIVE GATE FAILED: trial exception"; cat "$TRIAL/exception.txt"; exit 1; }
[ -s "$REWARD" ] || { echo "LIVE GATE FAILED: missing reward"; exit 1; }
[ -s "$TESTOUT" ] || { echo "LIVE GATE FAILED: missing verifier stdout"; exit 1; }

# tool_calls in trajectory.json exist only after Harbor's authoritative parser
# accepted a raw model response and executed it.
python3 - "$TRAJ" "$REWARD" "$TESTOUT" <<'PY'
import json,re,sys
traj,reward_path,testout_path=sys.argv[1:]
d=json.load(open(traj)); steps=d.get('steps',[])
actions=sum(len(s.get('tool_calls') or []) for s in steps)
reward=open(reward_path).read().strip(); testout=open(testout_path).read()
print(f'live steps={len(steps)} parsed_tool_calls={actions} reward={reward}')
assert actions > 0, 'Qwen3.8 produced no Harbor-parsed/executed command'
assert reward in {'0','1','0.0','1.0'}, f'invalid reward: {reward!r}'
assert re.search(r'\b(passed|failed)\b', testout, re.I), 'pytest did not execute'
for bad in ('uvx: command not found','pytest: command not found'):
    assert bad not in testout, f'verifier infrastructure error: {bad}'
print('QWEN38 FULL-STACK LIVE GATE GREEN')
PY
