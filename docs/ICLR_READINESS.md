# ICLR readiness ledger

Goal: every number in the paper traces to (a) on-policy data, (b) a certified
measurement, (c) a provenance-stamped result file, (d) a survived attack in
THREATS.md. This file tracks the distance to that bar. Companions: THREATS.md
(attack register), DECISIONS.md (conventions), REVIEW.md (reviewer paths),
ROADMAP.md (execution).

## The bar, per claim

| requirement | status | evidence |
|---|---|---|
| On-policy data only (no other-model traces) | ENFORCED | run_all.sh queue-onpolicy hard-fails without examples_onpolicy.json; Open-SWE files retired for measurement |
| Full real contexts (no fixed-size windows) | ENFORCED | prefetch_onpolicy keeps full history capped at 0.8 x native (or the largest servable window; see "window ceiling" below) |
| Deployment-parity sampling | DONE | max_new=10240 (harbor budget), temp 1.0 / top_p 1.0 (cited standard), per-model native formats |
| Parser = certified mirror of vLLM authority parsers | DONE | 5-format totality tests; qwen3_engine + glm47_moe verified in installed vllm 0.25; the Jul-24 XML-as-halt bug found, quantified (requants), tainted numbers retired |
| Call equality = BFCL AST matching | DONE | args[:60] truncation bug found and removed |
| D metric noise-honest | DONE | exp19 exact-vs-sampled (sampled tool-level TV is ~90% floor); all effects floor-referenced + paired permutation; BH correction on headline table |
| Task success = real TB2 Pass@1 through harbor | IN FLIGHT | GLM easy-25 complete (retry closing 9 infra rows); 35B all-89 relaunching behind window gate |
| Provenance in every result file | DONE (new) | common.save_result stamps git commit+dirty, versions, argv, SLURM job, input sha256 |
| Statistical tests paired + cited | DONE | experiments/stats.py (scipy-backed, equivalence-tested fallbacks) |
| Infra failures never read as model results | ENFORCED | verdict decomposition (ran/agent-timeout/env-infra); env-infra rows retried, never scored |

## Known limitations to STATE in the paper (not fix silently)

1. **Window ceiling**: 0.8 x native (209,715) faults in vllm 0.25 kernels
   (CUDA illegal access + cublas GemmEx fail on verified-clean GPUs - the
   int32-indexing wall). Runs use the largest servable window (search in
   flight; 163,840 candidate). State the ceiling and the stack version.
2. **Pass@1 comparability**: CompactionRL's ~27% class was measured at a 32k
   window (compaction pressure). Our larger-window numbers are an easier
   condition; report both conditions or annotate clearly.
3. **Halt semantics**: our halt is one-shot; deployed agents retry. Freeze
   law manifests as wasted turns, not literal stops.
4. **Single-model suite** until the H200 window (18-model coverage plan);
   current findings are existence proofs on Qwen3.5-35B + GLM-4.7-Flash.
5. **n_attempts=1** on TB2: Pass@1 over tasks, no per-task variance. Cheap
   to fix later with n_attempts=3 on solved-adjacent tasks if reviewers push.

## Remaining work, ordered

1. [auto] Window hunt verdict -> 4x TB2 35B jobs -> Pass@1 + trajectories.
2. [auto] Watcher -> examples_onpolicy.json -> 14-exp suite on-policy rerun.
3. GLM retry lands -> complete GLM easy-25 row (all 25 verdicts honest).
4. Aggregate: stats.py + make_figures.py over on-policy results; BH-corrected
   headline table; verdicts to AUDIT.md with job ids.
5. Group decision: vllm upgrade (to lift the window ceiling) vs document it.
6. H200 window: coverage matrix per H200_PLAN.md; the matched-pair rows
   (AgentWorld vs vanilla) are the strongest reviewer ammunition.
7. Camera-ready hygiene: pip freeze > environment lock per venv; pin model
   revisions (HF commit hashes) in eval configs.
