# ICLR readiness ledger

Last audited: 2026-08-18. The execution order is in `docs/ICLR_PLAN.md`; all
experiment dispositions are in `docs/EXPERIMENT_LEDGER.md`.

| Requirement | Status | Evidence / next action |
|---|---|---|
| On-policy evidence for main claims | DONE | 64-example own-agent dataset; powered N=25 exp4/8/17/20/21/23 |
| Full real contexts and deployment sampling | DONE | 2k–209k histories; temp=1/top_p=1; adaptive sampling |
| Parser/call equality certified | DONE | authority parser tests + BFCL AST matching; 144/144 test gate |
| Noise-honest behavioral metric | DONE | floors, paired tests, raw arrays; exp19 exact-TV audit |
| Logged-action grounding | DONE | exp8: full 0.65, keep-recent 0.66, summary 0.46 |
| Matched-budget verbatim-vs-rewrite test | DONE | exp23 N=25; verbatim tie, rewriting loses |
| Regime-dependence test | DONE | exp4/21 effects fail on-policy |
| TB2 verifier executes offline | DONE | real pytest smoke and reward file |
| Oracle can earn positive reward | **DONE 2026-08-18** | `tb2/oracle_smoke.sh`; reward=1 on break-filter-js-from-html |
| Final model preflight | IN PROGRESS | Qwen3.8-27B cache, then vLLM + parser gate |
| Competent-model baseline | OPEN | Qwen3.8-27B bf16 easy25 is Gate 1 |
| Live policy/outcome bridge | OPEN / CRITICAL | exp22 four-arm pilot, then powered row |
| Figures generated from committed artifacts | OPEN | after exp22 pilot |
| Full paper draft | OPEN | two-week sprint after Gate 1 |

## Claim readiness

| Claim | Status |
|---|---|
| Keep-recent at 25% is statistically indistinguishable from full behavior | READY, one-model/domain caveat |
| Verbatim policies tie; rewriting loses at matched budgets | READY, N=25 caveat |
| Off-policy compaction conclusions can fail to transfer on-policy | READY |
| Verbatim compaction improves task success | **NOT YET ESTABLISHED; exp22 required** |
| GLM has a universal format cliff | CUT: off-policy only |
| Skeleton+tail beats keep-recent | FORBIDDEN: p=0.39 tie |
| D-based training works | FORBIDDEN: clean null at tested dose |

## Required paper disclosures

1. Behavioral fidelity is not task optimality; outcome evidence is separate.
2. Usable N=25 is conditional on full-context acting >=0.5.
3. Absolute sampled D is floor-biased; comparisons are paired and
   floor-referenced.
4. D values are not comparable across experiments.
5. Scope is coding agents and one primary model/trace distribution.
6. Every pre-Aug-6 TB2 score is retracted due to verifier failure.
7. Infra failures are excluded/retried, never counted as task failures.

## Submission gate

Behavioral core ✅ · regime-dependence ✅ · verifier ✅ · oracle ✅ · competent
baseline ❌ · exp22 pilot ❌ · powered outcome row ❌ · figures/draft ❌.
