# Reviewing guide

How to check this work, at three depths. Written for a skeptical reader:
the fastest path to "where would this break?" for each claim.

## The 15-minute review (no GPU, no install)

1. Read `README.md` - the 7 findings, each with its numbers and experiment.
2. Read `docs/AUDIT.md` - the claims-and-confounds ledger. Every experiment
   has an entry stating what would invalidate it. If a finding in the README
   feels too clean, its caveats are there. Four results are explicitly
   flagged "will not stand" (exp6-coarse, exp13, exp15, exp7-GLM) and are
   NOT in the README findings table.
3. Spot-check one number: open `experiments/results/exp4_block_ablation_*.json`
   and confirm the halt rates in README finding 1 match the raw file. Every
   README number traces to a JSON here the same way.

## The 1-hour review (laptop, no GPU)

```bash
pip install -r requirements.txt
python tests/run_tests.py            # 68 checks: parser totality, metric
                                     # properties, compressor budgets, stats
cd experiments && bash run_all.sh cpu  # exp1 + exp2, pure arithmetic
```

Then read ONE experiment end to end - suggested: `exp4_block_ablation.py`
(the strongest finding). Check in order:
- the docstring states design + what was controlled
- budget fairness: every condition trims to the same token count
  (`random_trim` after the targeted deletion)
- the floor: full-vs-full resample measured per example, reported next to
  every effect
- the stats: paired permutation tests in `stats.py` (30 lines, no deps)

Code reading order if you want the whole library: `scaffold.py` (the task
environment) -> `behavior.py` (action sampling + total parser) ->
`compressors.py` -> `metrics.py` (D at three granularities) ->
`experiments/common.py`.

## The full reproduction (one A100/H100, approx 2 days)

```bash
HF_HOME=... python prefetch.py       # cache models (internet once)
cd experiments && bash run_all.sh gpu   # or: queue (Slurm)
```

Each script prints its summary table and writes a timestamped JSON with
raw per-example arrays - your run appends alongside ours, so drift is
directly visible. Terminal-Bench grounding: `tb2/README.md` (one script,
offline; note the measured 0% table there and what it means).

## Claim-by-claim: where each finding would break

| finding | weakest point to attack | our own check of it |
|---|---|---|
| 1 freeze law | is halting just parse failure? | parser is total (accepts 4 formats); halting = no tool call in ANY format; on-policy replication halt 0.66 vs 0.34 |
| 2 containment law | only 6 condition clusters, N=17 | flagged in AUDIT; needs continuous-rate replication (planned) |
| 3 format cliffs | is it just one weird model? | 4 models tested; effect present in 3, absent in dense-27B - claim is "model-specific", not universal |
| 4 tiny core | recency-biased metric favors it | yes - stated; the long-horizon test where this could fail is designed but not run |
| 5 select-not-train | training null could be underpowered | it is - approx 350 pairs, 4B student, LoRA; null scoped to that config in AUDIT |
| 6 summaries worst | off-policy summarizers unfairly bad? | on-policy + format-matched (summary_native) delivery retested; ordering held |
| 7 estimator noise | N=10 | methods-only claim; r itself not quoted |

## Questions we would ask (and our answers)

- **Is D just measuring formatting sensitivity?** Partly - that is finding 3.
  The content results (1, 2, 4) all hold format constant across conditions.
- **Does any of this predict task success?** Unproven. Grounding is
  agreement with logged actions (0.67-0.73), not solved tasks; the bf16
  Terminal-Bench run to fix this is queued (docs/MIGRATION.md).
- **One trace domain?** Yes - SWE-agent-style traces plus our own TB2
  trajectories. Second domain is on the open list.
- **Multiple comparisons?** Primary endpoints were pre-registered in
  script docstrings from exp11 onward (see exp21 for the pattern);
  earlier experiments are labeled exploratory in AUDIT.
- **Seeds?** exp4: 5 runs, 2 seeds. Most others single-seed at N=13-24
  examples with paired tests across examples. Multi-seed pass is on the
  open list.

## What is deliberately NOT on main

Slurm job scripts per cluster, figures, planning docs, one-off analysis
scripts: branch `research-archive` (identical history, nothing deleted).
Raw agent trajectories: `data/migration_payload.tar.gz` (153 files).
