# STATUS — the one living board (done / running / open)

Last updated: 2026-07-31, after merging the parallel Trillium session.
Roles: STATUS (this file) = roll-up truth · docs/ROADMAP.md = execution
phases & job plumbing · docs/ICLR_READINESS.md = rigor bar per claim ·
docs/AUDIT.md = append-only verdicts · THREATS.md = attack register ·
paper/OUTLINE.md = narrative. State changes go HERE + verdict to AUDIT.

## Governing decisions (group, 2026-07-30)

- **ON-POLICY ONLY**: Open-SWE (other-model) traces retired for
  measurement; all 14 measured experiments rerun on examples_onpolicy.json
  built from OUR TB2 trajectories. Existing requant numbers stay quotable
  as the off-policy record until the on-policy suite supersedes them.
- **Budget regime**: compaction fires at C−|h|<10240, ≤3/rollout; C=64k
  (paper-comparable). Giant windows were silently removing the studied
  phenomenon + hitting vllm's ~209k int32 kernel wall. Budget is the
  first-class per-model parameter.

## RUNNING (Trillium H100 — the on-policy suite, resubmitted 2026-08-04)

- **711526–711540 + 711586** — the full 15-exp on-policy suite (exp3–21 +
  exp23) on examples_onpolicy.json (64 full-context examples), 12h walls,
  from the execution clone $SCRATCH/dccb @ fc1d2c4
- History of this suite: Aug-1 submission (700347–60) died in 20s/job —
  Trillium sbatch wrapper word-split the --wrap string (9afb268 gotcha;
  run_all.sh queue now generates script files, NEVER --wrap). Aug-4 first
  resubmit (711507–21) died on missing `triton` in ENV-compress2@Trillium
  (env had never loaded a model on this cluster; triton 3.6.0+computecanada
  installed, fla import verified). Third submission is loading weights
  cleanly. exp6's first try was CANCELLED-by-0 at 6s (node-side); resubmitted.

## TB2 harvest (Phase 1, done 2026-08-01)

- GLM c64k easy-25: COMPLETED (15 trajectories). Qwen 35B: easy25 +
  shard_00 COMPLETED, shard_01/02 OOM-killed late (host RAM) — 59 Qwen
  trajectories total, 13/19 + 13/23 on the OOM shards. Watcher fired,
  built examples_onpolicy.json (64 examples), auto-queued the suite.
  Pass@1 rows for shard_01/02 are PARTIAL — rerun of missing tasks is an
  open item if full-89 coverage is needed for the paper table.

## DONE (recent; full history in AUDIT)

- All 4 headline requants survived hostile requantification (Jul 24–26)
- exp20b (containment ×3 rates); temp-1.0 robustness for freeze law
- Parser certified vs vLLM authority parsers; BFCL AST call-equality;
  provenance stamps in every result file; env lock files (ICLR_READINESS)
- TB2 harness end-to-end; 89/89 sifs; exp22 smoke GREEN (merged to main)
- Repo: STATUS/paper layer added; exp22-outcome branch merged+deleted;
  dead one-off job scripts removed; prefetch_onpolicy restored (upstream)

## OPEN — not covered by the auto chain (the real to-do)

- [ ] **exp4 structure-preserving ablation** (placeholder blocks) —
      threat 1.3; small diff on exp4, 1 H100 job after suite lands
- [ ] **GLM-as-executor temp-1.0 arm** — threat 3.2; the on-policy suite
      measures with 9B, so the GLM cliff still needs its own arm.
      Blocks publishing the 0.00/0.75 magnitude
- [ ] **E-B repo-level train/held-out split** (AUDIT MUST-fix; CPU)
- [ ] **exp2 sensitivity sweep** over GRPO sim assumptions (X.7; CPU, minutes)
- [ ] **Citation-graph walk** (paper/LITERATURE.md order; non-cluster
      machine — Semantic Scholar rate-limits from here)
- [ ] **exp22 H100 pilot** (2 arms × 1 model × 89 × 1) — the D→Pass@1
      bridge direction before spending the H200 budget
- [ ] exp10-free (free-running propagation) — if suite exp10 leaves the
      healing claim short
- [ ] Rewrite contradiction-prone claims in README findings table (#5, #6,
      freeze dual-quote) once on-policy suite numbers land

## AUTO (in the chain, do not duplicate — ROADMAP Phases 1–3)

Watcher → on-policy suite at deployment parity (temp 1.0, 10240,
certified parser) including powered exp14 (format cliffs), exp17/exp21 at
320×16k, exp6, exp20 — these supersede my earlier "Wave 1" powered-rerun
items. **exp23 (one-liner history + verbatim tail @25/50/75%, exp22
arm-B's offline twin; P1 pre-registered = hybrid > keep_recent @25%)
added to the queue-onpolicy list — rides the same auto chain.** Then Phase 3: competent-trace regeneration + exp8 with task
success + 64k arms (H100-80 clears the old OOM).

## DEFERRED — H200 window (docs/H200_PLAN.md)

18-model coverage, matched pairs, TP=8 giants, full exp22 matrix,
Phase-2 training round, OpenRouter rows.

## Submission gate (THREATS §7, restated against today)

requants ✅ · exp20b ✅ · temp-1.0 headline ✅ · on-policy suite 🔄 (auto) ·
GLM temp-1.0 executor arm ❌ · exp22 pilot ❌ · citation walk ❌
