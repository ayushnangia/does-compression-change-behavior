# STATUS — current truth only

Last updated: 2026-08-18. Detailed experiment disposition:
`docs/EXPERIMENT_LEDGER.md`. Execution plan: `docs/ICLR_PLAN.md`. Historical
verdicts and retractions: `docs/AUDIT.md`.

## Current state

- Slurm chain now uses durable dependencies: full Harbor live gate 805028 →
  valid Qwen3.8 baseline 805029 → on-policy data build 805030. GRPO is not
  auto-submitted; baseline competence and data power are manual gates.
- Repository/data: on-policy dataset (64 examples, 2k–209k tokens) and all
  Aug 4–6 powered result artifacts are committed.
- Test gate: **112/112** after exp22/exp24 lineage, budget, parser and
  generation-default guards.
- Paper policy: on-policy results only; off-policy appears only as evidence
  that evaluation regimes do not reliably transfer.

## What is established

1. **Verbatim recency is nearly free at 25%.** exp8 powered N=25: logged-action
   agreement full=0.65, keep-recent@25%=0.66, summary=0.46.
2. **A tiny extractive core remains useful.** exp17 powered N=25: agreement
   0.57 at 2%, 0.68 at 25%.
3. **Do not rewrite action history.** exp23 powered N=25: raw skeleton+tail and
   keep-recent statistically tie (pooled p=0.39); canonical rewriting loses
   about 13 points and summaries about 20.
4. **Off-policy laws did not transfer.** exp4 freeze law is null on-policy
   (0.35 vs 0.28 halts, p=0.78); exp21 wrapper effect is null/reversed (0.49
   vs 0.55, p=0.36).
5. **Containment is supporting, not a law.** exp20 powered rho=-0.23 on-policy,
   stronger than NLL (-0.12), but predictive only.
6. **Qwen format phenotype is null on-policy.** exp14 completed Aug 8, N=24:
   native=0.47 vs wrapper=0.48 at top_p=1. The GLM cliff remains off-policy
   and is not a paper headline.
7. **The 25/64 acting filter is not a long-context collapse.** Usable rates
   are 0.20/0.40/0.50/0.50 across <16k/16–32k/32–64k/64k+ bins; usable
   histories are longer on average, but not significantly (p=0.254).

## What failed or is retired

- exp6 powered rate curve timed out twice at 24h. Checkpointing saved N=16 but
  had no resume mechanism, so the second run repeated work. The partial floor
  is 0.45 [0.34,0.56] and policy ordering is unstable. **Retired from the
  critical path**; exp8/17/23 answer the central budget question more cleanly.
- exp3 is floor-dominated (0.68); exp12 has N=1; exp13 is confounded; exp15 is
  format-confounded; exp16/DPO is a clean null at this dose. None blocks the
  paper.
- Every TB2 score before Aug 6 is retracted: verifiers could not run offline.

## Validity gate

- Offline verifier smoke: **GREEN** (real pytest execution and reward file).
- Oracle/reference solution: **GREEN on 2026-08-18**. The reference solution
  for `break-filter-js-from-html` passed offline and wrote reward=1.
- Reproducible gate: `bash tb2/oracle_smoke.sh`.
- easy25 images: 25/25 rebaked with system-path uv caches.
- Remaining 64 images are stale; this does not block easy25 Gate 1, but they
  must be rebaked before an all-89 run.

## Sole next objective

**exp22: connect behavioral preservation to valid Terminal-Bench task
success.** Final lineage is `Qwen/Qwen3.8-27B` bf16 (user decision,
2026-08-18). After cache + vLLM/parser preflight, run its easy25 baseline. If
it yields at least two valid successes, run the four matched live policy arms:

A. keep recent verbatim;
B. raw action skeleton + verbatim tail;
C. stock Terminus-2 summary;
D. no compaction/truncation.

Arm B has been corrected from the rejected canonical one-liner design. A/B/C
now share a three-policy-compaction cap followed by the same fallback.

## Hard stop rule

No duplicate exp3–21 jobs, GLM cliff runs, full exp6 rerun, or broad model
matrix before the outcome gate. If the Qwen3.8-27B baseline is at floor,
switch model/scaffold; do not manufacture progress with more proxies.
A proper training follow-up is designed as **exp24 GRPO-D** with matched base,
SFT-best and DPO controls (`docs/RL_EXPERIMENT.md`); it starts only after exp22
establishes a non-floor outcome evaluator.

## Submission gate

- on-policy behavioral evidence: **DONE**;
- off-policy-regime warning: **DONE**;
- verifier + oracle validity: **DONE**;
- Qwen3.8-27B cache: **DONE**, snapshot `1d4bf0f...`, 52 GiB;
- full-window serving: **DONE** (77,824 tokens, 51.1 GiB weights, job 803407);
- raw-interface preflights 803407/803548: **retired as invalid proxies**;
- full Harbor live gate: job 805028;
- competent easy25 baseline: job 805029, Slurm `afterok:805028`;
- Qwen3.8 on-policy exp24 data build: job 805030, `afterok:805029`;
- exp22 four-arm pilot: **OPEN** after baseline competence check;
- powered outcome row: **OPEN**;
- paper draft/figures: **OPEN**.
