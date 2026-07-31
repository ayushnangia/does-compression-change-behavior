# STATUS — the one living board (done / not done / running)

Last updated: 2026-07-31. Convention: this file is the ONLY current-state
document. AUDIT.md is append-only history; THREATS.md is the attack
register; paper/OUTLINE.md is the narrative. When state changes, update
HERE and append the verdict to AUDIT.md.

## Headline findings (paper §4–5)

| finding | number (requant standard) | publishable? |
|---|---|---|
| Freeze law (exp4) | halts 0.31 vs 0.10 (3.1x) @0.7; 0.42 vs 0.21 @1.0; obs free | YES (structure-preserving ablation pending strengthens vs threat 1.3) |
| Containment law (exp20b) | rho −0.38/−0.44/−0.45, N=23 x3 rates | YES (predictive-only phrasing) |
| GLM format cliff (exp14/requant) | 0.00 vs 0.75 acting, N=24 | **NO — temp-1.0 arm required (threat 3.2)** |
| Wrapper effect (exp21) | +18pts p=0.023 @0.7; +12pts p=0.121 @1.0 N=18 | **NO at the cited standard — powered N~64 @1.0 required** |
| Tiny core (exp17/21) | 0.59 @2% (330 tok); ~870-tok canonical history | YES (one model caveat) |
| Selection works (exp11/T4) | p=0.0004; 16k replication non-overlapping CIs | YES (circularity disclosed; exp22 is external fix) |
| Training null (expB/T4) | 0.73 vs 0.75, p=0.345 | YES as scoped null only ("at this dose") |
| Summaries worst (exp6/9) | — | WEAKENED: production comparison = exp22 arm C only; phrase as "summarizers we implemented" until then |

## DONE (do not redo)

- Hostile requant of all 4 headlines (parser certified vs 5 formats,
  10240 budget) — all survived, magnitudes re-quoted (AUDIT Jul 24–26)
- exp20b 3-rate containment replication; NLL claim honestly revised
- Temp-1.0 robustness for freeze law + observation asymmetry (Jul 27)
- BH correction over m=6 primaries — 3 headlines pass
- exp19 exact-estimator validation (floor-referencing load-bearing)
- Metric critique triage: 8/11 limitations FIXED in code, unit-tested
- TB2 harness end-to-end (vLLM+harbor+Apptainer offline); ENV-vllm3;
  all 89 sifs baked; exp22 smoke GREEN (policy fired 15x live)
- paper/ scaffolding: RQ, RELATED, OUTLINE, LITERATURE

## RUNNING (Trillium H100, check `sq`)

- TB2 bf16 chain: jobs 696621, 696661–696664 (5x eval_tb2.sh) —
  competent-model Pass@1 + on-policy trajectories. Clean rerun after
  683764 NODE_FAIL. ETA ~1 day.

## NOT DONE — Wave 0 (CPU/login, start anytime, no queue)

- [ ] E-B repo-level train/held-out split (AUDIT MUST-fix)
- [ ] exp2 sensitivity sweep over GRPO sim assumptions (threat X.7)
- [ ] Parser certification harness: finish + declare done (threat X.6)
- [ ] Citation-graph walk (paper/LITERATURE.md order; non-cluster machine
      for Semantic Scholar)
- [ ] Rewrite 3 contradiction-prone claims (README #5/#6, freeze dual-quote)
- [ ] PYTHONUNBUFFERED in job scripts (cosmetic)

## NOT DONE — Wave 1 (single-H100 jobs, submit as GPUs free)

- [ ] exp4 structure-preserving ablation (placeholder blocks) — threat 1.3
- [ ] GLM temp-1.0 arm — UNBLOCKS the cliff claim; split per-condition
      jobs (GLM eats 12h walls at deployment budget)
- [ ] Powered wrapper run @1.0, N~64, split 3–4 parallel jobs — UNBLOCKS
      wrapper claim at the standard
- [ ] exp6-coarse @N=48 — settles the on-policy flip (README #6 conflict)
- [ ] 32k/64k arms via vllm_scorer (OOM'd on A100-40; H100-80 clears)
- [ ] exp10-free (free-running propagation; no env needed)

## NOT DONE — Wave 2 (after TB2 harvest)

- [ ] prefetch_onpolicy from bf16 trajectories -> rerun exp4 + exp17 on
      competent-agent traces (half-answers threat 4.1)
- [ ] exp22 H100 pilot: 2 arms (A vs C) x 1 model x 89 x 1 — direction of
      the D->Pass@1 bridge + compaction-frequency calibration before H200
- [ ] exp15 format-matched redo (summary_native)
- [ ] Multi-seed pass exp4/exp6/exp9

## DEFERRED — H200 window (docs/H200_PLAN.md, unchanged)

9-lineage survey, TP=8 giants, full exp22 matrix (4 arms x 2 models x 2
runs), Phase-2 training round, OpenRouter frontier rows.

## Submission gate (THREATS §7)

requants ✅ · exp20b ✅ · temp-1.0 headline ✅ · GLM temp-1.0 ❌ ·
powered wrapper @1.0 ❌ · exp22 (pilot minimum) ❌
