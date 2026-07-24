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

## Measured results on this harness (Narval A100-40GB)

| model | subset | Pass@1 |
|---|---|---|
| Qwen3.5-9B bf16 | full 89 | 0/53 started |
| Qwen3.5-35B-A3B GPTQ-Int4 | full 89 | 0/69 |
| Qwen3.5-35B-A3B GPTQ-Int4, 4x timeout | easy 25 | 0/21 |

Robust zeros: quantized/small models are too weak for TB2 (consistent with
the CompactionRL paper's findings; their 35B bf16 scores approx 27%). The 35B
bf16 needs more than 40GB per GPU or working multi-GPU serving - on this
cluster vllm 0.25 hangs at multi-GPU MoE engine init (jobs 66203598 et al.),
which is why the H100 migration exists (MIGRATION.md). The trajectories are
still used as on-policy behavioral data regardless of task success.
