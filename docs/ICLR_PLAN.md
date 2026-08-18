# ICLR recovery plan — one objective, hard gates

Last updated: 2026-08-18.

## The next objective

**Establish whether the on-policy behavioral result transfers to real task
success.** Run exp22 on Terminal-Bench with four live arms at matched
compaction triggers and budgets:

A. keep recent verbatim;
B. raw action skeleton + verbatim tail;
C. stock Terminus-2 summary;
D. no compaction / truncation baseline.

Primary endpoint: valid Pass@1. Secondary endpoints: number of compaction
events, post-compaction acting/stall rate, and offline D at those events.
The policy hypothesis is **A/B > C**; A vs B is explicitly a tie hypothesis,
not a winner claim. D is a system baseline, not assumed to rank last.

Duplicate behavioral proxy work is frozen until this bridge has a result.
The next major experiment after a non-floor Gate 1 is **exp24 GRPO-D**, a
proper training comparison against base, SFT-best and DPO; see
`docs/RL_EXPERIMENT.md`.

## Why progress stopped

This was not a lack of experiments; it was a failure to close the loop.

1. **No durable controller after Aug 9.** The queue has been empty since then,
   while STATUS still says jobs are running/queued.
2. **The TB2 auto-chain was brittle.** It raced the easy25 rebake, saw 88/89
   images, exited, and never retried. The full rebake supervisor used `pgrep -f`
   patterns that can match the supervisor itself. No Slurm dependency or durable
   completion marker existed.
3. **Checkpointing was not resumption.** exp6 saved partial JSON but restarted
   from example zero. Two 24-hour H100 jobs reached the same example and timed
   out. The N=16 checkpoint is recoverable, but finishing exp6 is not the paper
   bottleneck.
4. **Completed results were not harvested.** exp14 completed Aug 8 (N=24) but
   remained scratch-only; it is a null on-policy (native 0.47, wrapper 0.48),
   not an in-flight format result.
5. **The verifier invalidated all earlier Pass@1.** Fixing it took two rounds.
   The offline smoke was green, but no oracle had been run, so the validity gate
   was still incomplete.
6. **exp22 drifted behind the science.** Its arm B still used canonical
   one-liners even after exp23 rejected rewriting; cap handling also differed
   across arms.
7. **Too many objectives remained alive.** Rate curves, GLM cliffs, DPO,
   18-model coverage, citation work, and outcome evaluation competed without a
   stop rule. More proxy measurements could not answer the reviewer’s main
   question.
8. **Project truth fragmented.** STATUS, ROADMAP, ICLR_READINESS, OUTLINE and
   slides contradicted the Aug 6–9 results. This plan + STATUS + the experiment
   ledger now supersede stale queue language.

## Gates and execution order

### Gate 0 — measurement validity: GREEN

- Offline verifier executes real pytest with no dependency downloads: green.
- Reference `solution/solve.sh` passes and writes reward=1: **green on
  `break-filter-js-from-html`, 2026-08-18**.
- Reproducible command: `bash tb2/oracle_smoke.sh`.

### Gate 1 — competent baseline

Cache and preflight `Qwen/Qwen3.8-27B` bf16, then run stock Terminus-2 on
easy25 once using rebaked images. The preflight must prove current vLLM can
serve the architecture and that real serialized contexts yield actions parsed
by the certified Qwen parser.

- **GO:** at least 2 valid successes and no verifier/infra failures. Proceed to
  the four-arm pilot on a fixed task subset.
- **STOP/SWITCH MODEL:** 0–1 successes. Do not compare policies on a floor.
  Move to the strongest available H200 model/scaffold and repeat Gate
  1; do not restart behavioral proxy experiments.
- Every trial must be classified `valid reward`, `agent timeout`, or `infra`.
  Infra rows are retried and excluded, never scored as failures.

### Gate 2 — exp22 pilot

Run four arms on the same easy25 tasks, one attempt each, randomized arm order.
Required logging before launch:

- policy name and git commit;
- exact input/token budget and trigger;
- compaction count and locations;
- valid verifier marker;
- task reward and failure class.

Analysis:

- report Pass@1 with Wilson intervals;
- paired task-level arm differences (McNemar/exact paired permutation);
- report the subset that actually compacted separately;
- never infer an arm ordering from fewer than two successes per compared arm.

**GO to powered run:** pipeline clean and at least one arm has enough successes
for a non-floor comparison. Otherwise switch model/scaffold.

### Gate 3 — minimum submission experiment

One competent model, all valid offline-solvable TB2 tasks, four arms, two seeds
if compute permits. Freeze policies before this run. The main figure links:

`policy -> behavioral preservation at compaction -> Pass@1`.

A second model is valuable only after this complete row exists.

## Two-week sprint

| Window | Deliverable | Exit criterion |
|---|---|---|
| D0–D1 | Oracle + baseline | reward=1 oracle committed; baseline job submitted and monitored |
| D2–D4 | Baseline harvest | honest valid-success count; GO or model-switch decision |
| D4–D7 | exp22 four-arm pilot | clean per-task table and compaction-event logs |
| D7–D10 | Powered row or stronger-model Gate 1 | at least one interpretable outcome comparison |
| D10–D12 | Figures + statistics | three figures regenerated from committed artifacts |
| D12–D14 | Full paper draft | intro/method/results/limitations complete; no placeholder claims |

## Minimum viable ICLR paper

1. **Instrument:** behavioral distortion with noise floors and logged-action
   grounding (exp8/19).
2. **On-policy law:** verbatim extractive context preserves behavior; rewriting
   loses it at matched budgets (exp17/20/21/23).
3. **Methods result:** off-policy laws can disappear on the evaluated agent’s
   own trajectories (exp4/21).
4. **Outcome bridge:** exp22. Without this, submit only if explicitly framed as
   a behavioral-measurement paper; do not imply task utility.

## Explicitly deferred

- full exp6 D(R) completion;
- GLM on-policy format-cliff arm;
- exp24 GRPO-D main run (designed now; starts after a non-floor outcome gate);
- 18-model H200 matrix;
- second agent domain;
- structure-preserving exp4 ablation.

These are reviewer-response or follow-up experiments, not prerequisites for
learning whether the current paper has a defensible main result.
