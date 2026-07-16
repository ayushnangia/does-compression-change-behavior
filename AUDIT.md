# Claims audit — evidence, confounds, rigor status

Last updated after the soundness pass. Rules: every claim lists its threats;
a claim is "presentable" only when its threats are either controlled or
explicitly stated next to it. Primary endpoints are fixed BEFORE looking at
new runs.

## Claim table

| # | Claim | Evidence | Threats & status |
|---|-------|----------|------------------|
| C1 | CompactionRL trains early summaries at a fraction of late ones (13.5% at chain depth 4) | exp1 (exact arithmetic from their hyperparameters) | none — it is their own equation. Presentable. |
| C2 | Segments-as-samples distorts GRPO (3.2x gradient mass; +0.37 advantage inflation) | exp2 (Monte Carlo) | simulation assumptions (K distribution, success rates) are stipulated — presentable as "under stated assumptions". |
| C3 | Behavioral damage is front-loaded across the chain; credit is back-loaded | exp7 + exp1 | (a) ceiling effect on label-level TV — mitigated: coarse tool-level metric added for reruns; acting-rate trend (0.53->0.46) is ceiling-free; (b) iterated summarization is a lower-bound proxy (stated); exp10 (running) adds the realistic-continuation version. Presentable with caveats. |
| C4 | Blocks carry very different behavioral value per token | exp4 v2 (corrected, queued) | v1 was MISLABELED (drop_observations was a no-op; drop_reasoning removed reasoning+observations). v1 numbers support only the 2-way claim (tool calls precious vs pooled content droppable). 3-way claim awaits v2. DO NOT quote v1 labels. |
| C5 | Interface format alone moves behavior at constant information | exp5 | wrappers differ slightly in token count (small, stated); single model. NEW: wrapper_only control added to exp9 so format-cost has an explicit zero point in reruns. Presentable with caveats. |
| C6 | The full-context SSL target is stable at moderate lengths | exp3 v1 | CONFOUND: truncation removes real information — "drift" at 1k mixes instability with legitimate difference. v2 (queued) adds the padding control (same info, padded to max length) that separates them. Quote v1 only as "prefix-depth sensitivity". |
| C7 | D is grounded in external reality (logged actions) | exp8 (running) | "reality" = the 122B generator's policy, not optimality — frame as external, non-self reference. |
| C8 | Deployed compaction prompts differ measurably; D can rank them | exp9 (running) | length confound now recorded per policy; wrapper_only zero point in reruns. |
| C9 | One compaction's damage persists/heals through real continuation | exp10 (running) | teacher-forcing holds the continuation to the ORIGINAL branch — measures divergence under the real trajectory, not free-running divergence (stated in docstring). Clean causal design otherwise. |
| C10 | D works as a selection signal (best-of-N) | exp11 (queued) | winner's curse FIXED (fresh-seed rescoring); primary endpoint pre-registered = downstream D (selection-independent machinery); immediate D demoted to confirmatory. |
| C11 | D works as a training signal (DPO compressor) | E-B chain (queued) | pairs gap-filtered (>=0.25); tiny-N proof-of-mechanism only; co-adaptation risk checked by exp12; index-based (not repo-based) train/held-out split — MUST upgrade before any writeup. |
| C12 | Summary quality is (or isn't) portable across executors | exp12 (queued) | tokenizer-identity guard in code; two models same family — GLM run extends. |

## Cross-cutting threats (apply to all GPU results)

1. **N=14-24, mostly one seed** — CIs now printed (bootstrap) and raw
   per-example arrays saved in results for post-hoc tests. scale_job adds
   N=48 and a second seed for exp4.
2. **Paired design** — all conditions share examples and seeds; the correct
   tests (paired permutation) are now implemented in `stats.py` and wired
   into exp4/exp11 as primary-endpoint tests.
3. **Selection filter** (full-context acting >= 0.5): population =
   "decision points where the model acts confidently." All claims are
   conditional on it. State it.
4. **TV plug-in bias** at 8 samples: inflates absolute distances; cancels in
   like-vs-like contrasts; never quote absolute D without the floor.
5. **Off-policy traces** (written by qwen35-122B): within-experiment
   contrasts valid; external validity pending on-policy replication.
6. **Multiple comparisons**: many cells across 12 experiments. Discipline:
   one pre-registered primary endpoint per experiment (listed above);
   everything else is exploratory.
7. **Wrapper discontinuity**: compressed contexts get a compaction-note turn
   + mid-turn recent fragment that full contexts lack. Now measurable via
   exp9's wrapper_only control.
8. **TB2 pass@1 variance**: n_attempts=1 at temperature 1.0 over 89 tasks
   gives ~±4-5 point noise; do not over-read small model deltas.

## Rigor upgrades applied this pass

- `experiments/stats.py`: bootstrap CIs + paired permutation tests.
- Raw per-example arrays saved in exp3/exp4/exp6/exp11/exp12 results.
- CIs printed in exp4/exp11 tables; paired p-values for primary endpoints.
- Pre-registered primary endpoints (see table).
- wrapper_only control (exp9), padding control (exp3), role-aware blocks
  (exp4/exp6), winner's-curse guard + fresh rescoring (exp11), DPO pair gap
  filter, coarse tool-level metric (ceiling), summary-wrapper budget fix.

## Post-results flags (added after the full harvest)

- **GLM exp7 verdict (verified by debug_glm_summary): NOT a parser bug —
  reclassified from quarantine to finding.** GLM-4.7-Flash writes coherent
  summaries but its CONTINUATIONS from summary-wrapped contexts are
  incoherent token salad (raw dumps in glm_check_*.out), while the same
  model in the same job is coherent on trace-format contexts (exp4@GLM).
  Reading: catastrophic interface fragility — the compaction-note format is
  far enough out-of-distribution for GLM that its next-token distribution
  degenerates. This is the extreme version of exp5's interface claim,
  replicated across families. Caveat: our sampling uses top_p=1.0/top_k=0
  (full distribution), which amplifies OOD degeneration; production-style
  truncated sampling would look less dramatic — rerun with top_p=0.95
  before quoting numbers, but the qualitative contrast (Qwen robust, GLM
  not) stands.
- exp11 ran pre-rigor-patch, so no p-values in its JSON; recovered
  per-example pairs from its log: selection-time paired p = 0.0004 (n=19).
  Quote the FRESH-scored effect (0.612 vs 0.790) with this p as supporting.
- exp13's scaffold mode: menu itself advertises retrieval -> manifest value
  masked (stated at results time; retest designed).
- **TB2 Pass@1 = 0 is now a robust negative**: full 89 tasks at default
  15-min budgets (0/53 and 0/69) AND the 25 easiest tasks at 4x timeout
  (0/21; 14 genuine completions scored 0). Qwen3.5-35B-A3B-GPTQ-int4 under
  the Terminus-2 scaffold does not solve TB2 tasks in our harness. Gap to
  the paper's 27% plausibly: int4 vs bf16 (3B-active MoE quantizes badly),
  Terminus-KIRA vs Terminus-2, compaction-enabled 64k x 4 budgets, thinking
  overhead. Consequence: task-level grounding of compression deltas is not
  measurable at this model scale; behavioral grounding stands on exp8
  (trace-logged action agreement) instead. Silver lining: the completed
  runs produced ON-POLICY trajectories for the next prefetch round.

## E-B round-1 post-mortem (CORRECTED verdict)

Earlier reporting said "the DPO adapter is worse than base." The debugging
loop found the honest verdict is "**DPO round 1 did not train, and the eval
could not have detected it either way**":
- rewards/accuracies 0.367 (below chance), margins 0.015, loss ~ln(2):
  the adapter is a noise perturbation, not a trained compressor.
- ROOT CAUSE 1 (design): pairs were 9B-written but the 4B was trained —
  off-policy preference data; the trainee's logps over another model's text
  carry no usable signal. FIXED: exp11 --summarizer-model (trainee writes
  candidates; 9B executor still scores).
- ROOT CAUSE 2: ~4 optimizer steps on 11 pairs. Fixed by T4's 200-pair
  round (~75 steps).
- ROOT CAUSE 3: evaluate_compressor had no raw arrays, single summary draw,
  no CI/paired test — the reported 0.69-vs-0.83 gap is unverified. FIXED:
  multi-draw summaries, CIs, paired p, raw saved.
Do not quote round-1 numbers except as an infrastructure lesson.

## Scale-up round findings (July 16)

- **exp6-coarse resolves the on-policy flattening: it WAS a metric ceiling.**
  At tool-level granularity the on-policy D(R) landscape un-flattens, and the
  ordering CHANGES: block_aware becomes the best curve at R>=0.5 (coarse
  change 0.23-0.24, acting 0.66-0.70) while keep_recent loses its off-policy
  dominance (acting 0.39-0.53). Revised claim: truncation-beats-summary is
  the OFF-policy 4k story; ON-policy, the measured block-aware design wins.
  Status: preliminary (N=13). First run saved no coarse per-example arrays
  (payload gap, fixed); CI/paired-test rerun submitted (65709063). Do not
  quote effect sizes until it lands.
- exp4@16k on-policy: halting effect replicates (none 0.42 vs 0.16); change
  column ceiling-limited as expected.
- exp15 v1 OOM'd at 24k reconstructions x8 samples on A100-40 (HF path);
  refs capped to 16k (consistent with the program) + allocator hint;
  rerun 65708856.

## Still open (known, accepted, scheduled)

- Repo-level train/held-out split for E-B (before any writeup).
- On-policy trace replication.
- Multi-seed everything (only exp4 has 2 seeds so far).
- Second family (GLM) results pending.
- PYTHONUNBUFFERED in job scripts (cosmetic: live progress in .out files).
