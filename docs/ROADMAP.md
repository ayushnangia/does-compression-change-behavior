# Experiment roadmap: what runs where, on what, and why it's rigorous

Updated 2026-07-30 (Trillium). Companion to AUDIT.md (verdicts), THREATS.md
(vulnerability register), DECISIONS.md (conventions), H200_PLAN.md (coverage).

## Rigor invariants (checked, not assumed)

- **Real contexts**: examples_16k = 16,384 real context tokens + 1,024 recent
  per example, cut from real traces with logged ground-truth actions and
  future segments attached (32k/64k files scale accordingly; verified by
  direct inspection of context_ids lengths).
- **Real compaction**: compressors.py operates on token ids - model-generated
  summaries (budget 256), keep_recent, paraphrase, summary_native, pointer,
  hallucinator. The pointer/hallucinator pair is the metric-honesty control.
- **Deployment-parity harness**: samples=8, max_new=10240 (Terminus budget),
  temp 1.0 / top_p 1.0 (cited standard), 5-format parser certified against
  vLLM authority parsers (see DECISIONS). Gate: tests/run_tests.py (69) before
  ANY submission.
- **Release precision only**: bf16 or vendor-native. No community quants
  (int4 scored 0 on TB2 - tb2/README.md).

## Phase 0 - infrastructure (DONE, Trillium)

All 7 cluster gotchas found and encoded in scripts/MIGRATION.md: gcc-before-
cuda, virtualenv --no-download, no stdbuf around vllm (GLIBC_ABI_DT_RELR),
Qwen3.5 Mamba cache (max-num-seqs 128), --export=NONE, read-only $HOME on
compute, sifs must be BUILT with bake.def.tpl (harbor server inside).
89/89 task sifs baked+verified; 35B/9B/GLM-Flash cached in $SCRATCH/hf.

## Phase 1 - task success axis (TB2 Pass@1, IN FLIGHT)

| run | model | job | status |
|---|---|---|---|
| easy-25, 4x timeouts | Qwen3.5-35B-A3B bf16, 1xH100 | 694202 | RUNNING |
| easy-25 | Qwen3.5-9B bf16 (scale-contrast row) | 694247 | RUNNING |
| easy-25 | GLM-4.7-Flash 30B-A3B (lineage 2) | waiter -> smoke -> eval | download |
| easy-25 | GLM-4.5-Air 106B TP=4 (oracle scale) | DEFERRED by group | - |

Success feeds Phase 3: solved trajectories -> on-policy examples
(prefetch_onpolicy.py) -> exp8 grounding with task success available.

## Phase 2 - behavioral suite at deployment conventions (QUEUED, 14 jobs)

1xH100 x <=12h each, Qwen3.5-9B measuring model, fixed parser, temp 1.0.
Jobs 694255-694269 under def-rgrosse (use SBATCH_ACCOUNT; def-zhijing
fairshare is exhausted - 0.085 vs 0.262).

| exp | headline claim | data (real tokens) |
|---|---|---|
| exp3 | target length-stability | 64 x 4k |
| exp4 | BLOCK LAW: freezes trace to specific deleted blocks | 64 x 4k |
| exp5 | format vs content | 64 x 4k |
| exp6 | rate-distortion of compression | 64 x 4k |
| exp7 | compaction-chain damage | 64 x 4k |
| exp8 | grounding vs real trajectories | 64 on-policy |
| exp9 | summary-policy comparison | 64 x 16k |
| exp10 | error propagation | 64 x 16k |
| exp11 | SELECTION WORKS (best-of-N summaries) | 64 x 16k |
| exp12 | portability | 64 x 4k |
| exp14 | FORMAT CLIFFS (interface fragility) | 64 x 16k |
| exp17 | MINIMAL CORE | 320 x 16k |
| exp20 | CONTAINMENT LAW (OOD bridge) | 64 x 4k |
| exp21 | ONE-LINERS (canonical skeleton) | 320 x 16k |

exp1/exp2 are analytic (rerun on login node, outputs reproduced).
exp13/15/16/18/19/22 are done/superseded/coauthor-in-flight (AUDIT.md).

## Phase 3 - on-policy regeneration (BLOCKED on Phase 1 results)

1. If 35B produces solved easy-25 tasks: regenerate 16k/32k on-policy example
   sets from COMPETENT trajectories (prefetch_onpolicy.py, login node).
2. Rerun exp8 grounding with task success as the outcome variable.
3. 64k arms of exp4/exp8 (examples_64k.json ready; OOM'd on A100-40,
   fits H100-80). ~2 jobs x 12h.

## Phase 4 - H200 window (PLANNED, docs/H200_PLAN.md)

18-model coverage (9 local lineages + API tier), 2,700 GPUh budget,
per-model pre-flight gate. Priority order if window shrinks: AgentWorld
matched pair -> gpt-oss-120b -> Laguna pair -> Leanstral -> Nemotron-75B ->
TP=8 giants by TB2 score. Phase-2 DPO retraining (full-FT 9B, 2k pairs)
rides this window at the cited eval standard.

## Resource ledger (Trillium, def-rgrosse)

| block | GPU-hours (est) |
|---|---|
| TB2 x 3 models (Phase 1) | ~40 |
| behavioral suite x 14 (Phase 2) | ~100 (most exps << 12h cap) |
| Phase 3 (on-policy + 64k arms) | ~30 |
| total this cluster | ~170 GPUh vs ~250k GPUh/yr allocation share |

## Operational rules

- Everything submits with SBATCH_ACCOUNT=def-rgrosse until def-zhijing
  fairshare recovers.
- Any new model: pre-flight gate first (parser-in-installed-vllm check
  passed for glm47_moe + qwen3_engine on vllm 0.25).
- Any failure: postmortem to MIGRATION.md gotchas before resubmission.
- Results land in experiments/results/ (suite) and $SCRATCH/tb2/jobs/ (TB2);
  verdicts go to AUDIT.md with job ids.
