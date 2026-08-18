# Terminal-Bench 2.0 harness (offline, Slurm + Apptainer)

One script evaluates any locally cached model on TB2 with the Terminus 2
agent, fully offline: vLLM serves on the node, harbor drives the agent,
task environments run from pre-baked Apptainer images.

## One-time setup (login node)

```bash
# 1. cache the model
HF_HOME=$SCRATCH/hf hf download Qwen/Qwen3.5-9B

# 2. download the task set (needs internet -> login node)
harbor datasets download terminal-bench-2 -o $SCRATCH/tb2/terminal-bench

# 3. bake all 89 task images to .sif (approx 2h, resumable)
bash bake_all_sifs.sh
```

## Run

```bash
sbatch eval_tb2.sh Qwen/Qwen3.5-9B qwen35-9b            # full 89 tasks
sbatch eval_tb2.sh Qwen/Qwen3.5-9B qwen35-9b 1 easy25   # 25 easiest
sbatch eval_tb2.sh zai-org/GLM-4.7-Flash glm47 2        # tensor-parallel 2
```

Results land in `$SCRATCH/tb2/jobs/tb2-<served-name>/` - per-task verdicts
plus full `agent/trajectory.json` files (these trajectories are the on-policy
data source for the behavioral experiments; see `prefetch_onpolicy.py`).

## Gotchas learned the hard way (all encoded in the script)

- `hosted_vllm/<name>` needs exactly one `/` -> always set `--served-model-name`
- litellm phones home for a pricing json -> `LITELLM_LOCAL_MODEL_COST_MAP=True`
- `model_info` block is required or litellm rejects the unknown model
- task subsets must be a `tasks:` list in the YAML (repeated `-p` flags do not accumulate)
- Apptainer needs `APPTAINER_TMPDIR=$SLURM_TMPDIR` or it fills the home quota

## Validity status and retraction

**Every Pass@1 number produced before 2026-08-06 is retracted.** The task
verifiers attempted to install uv/pytest from the internet at scoring time;
compute nodes are air-gapped, so tests never executed and rewards defaulted
to zero. Those rows measured infrastructure failure, not model capability.
The trajectories remain valid behavioral data because agent execution was
unaffected.

The repaired images install uv in `/usr/local/bin`, prewarm task-specific
caches under `/opt`, and force offline resolution. Two gates now apply:

```bash
# Must execute real tests and write a reward.
# Must then apply a known-good solution and obtain reward=1.
bash oracle_smoke.sh
```

Both gates are green on `break-filter-js-from-html` as of 2026-08-18. The
35B bf16 easy25 baseline is the first capability measurement to run after
this repair. Never quote a task score without a verifier-clean marker; retry
infrastructure failures rather than counting them as model failures.
