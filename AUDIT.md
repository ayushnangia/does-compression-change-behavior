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

## Final harvest (July 16, second pass)

- **exp6-coarse WITH CIs**: on-policy at R=0.5, block_aware coarse-D 0.23
  [0.13,0.34] vs keep_recent 0.47 [0.35,0.62] (paired p=0.074) and summary
  0.41 (p=0.095). Direction consistent at R=0.75, weaker. Verdict: the
  ceiling explanation confirmed; block-aware-best-on-policy is REAL but
  marginal at N=13. Quote as "suggestive (p~0.07-0.10)"; N=48 rerun would
  settle it.
- **exp15 v2's 0.00-acting anomaly was OUR construction artifact**: post
  segments were serialized separately, so wrapper-based variants began with
  a fresh <agent_trace> document header and no trace body; the model treated
  them as a task starting. v1 of the geometry has the same flaw, so NO prior
  exp15 numbers are quotable except keep_recent-style trends. Fixed (single
  serialization, char-offset slicing, verified headerless); rerun 65729683.
- **T4 phase 1 SUCCEEDED at scale**: 177 usable examples at 16k, floor 0.253
  (marginalized target), select-by-D fresh 0.67 [0.64,0.69] vs random 0.72
  [0.69,0.74] — E-A replicates at 16k with non-overlapping CIs. 86
  gap-filtered pairs written. Phase 2 OOM was allocator FRAGMENTATION
  (15.3GB reserved-unallocated); expandable_segments added; resume job
  65729682 (train+eval only, pairs preserved).

## exp15 v3 verdict + T4 diagnosis (July 16, evening)

- **exp15 v3 (clean geometry, verified headerless): the wrapper-condition
  silence PERSISTS** (production and wrapper_only acting 0.00; block_aware
  0.27; keep_recent 0.76). Reclassified from artifact to format effect: on
  on-policy Terminus-format contexts, note-style injection suppresses
  parseable actions for the 9B executor, while trace-shaped compression
  acts. Consequence: the production-vs-designed summary comparison is
  CONFOUNDED BY DELIVERY FORMAT; do not quote it. Quotable: keep_recent
  agrees 0.74 with the real logged action at genuine window-exhaustion
  boundaries (N=19). block_aware at 0.24 here (reversal vs exp6-coarse;
  different regime). Follow-up designed: deliver the production summary in
  trace shape (format-matched) before comparing content quality.
- **T4 resume failed with the identical fragmentation OOM because torch
  2.9.1 reads PYTORCH_CUDA_ALLOC_CONF, not the newer PYTORCH_ALLOC_CONF**
  that was set (venv torch was downgraded by the causal-conv1d wheel).
  Both names now set; resubmitted (65785915).

## Limitations of D and their dispositions (the metric critique, resolved)

| # | issue | disposition |
|---|-------|-------------|
| 1 | parser-mediated (misses inflate D) | FIXED: parse_diagnosis classifies every non-action (prose / toolish_unparsed / think_runaway / empty); toolish_unparsed > 0 in any cell = parser bug flag, not a finding |
| 2 | no semantic action equivalence | MITIGATED: verb-level granularity added (tool + command head); full semantic equivalence remains open |
| 3 | consequence-blind | STRUCTURAL: requires task outcomes; bf16 TB2 run is the fix-in-progress |
| 4 | single-step myopia | BY DESIGN + documented rule: never quote D@0 without exp10-style D@k context |
| 5 | entropy-dependent small-sample bias | FIXED: debiased_change — per-comparison permutation null gives a debiased effect size and an exact p, immune to entropy differences |
| 6 | granularity arbitrariness | FIXED as a standard: action_change_all reports label/verb/tool together; single-granularity conclusions are artifacts |
| 7 | ceiling compression in noisy regimes | MITIGATED by 5+6; floors > 0.5 flagged as weak-regime |
| 8 | penalizes beneficial divergence | FIXED as an additional metric: harm_score counts only halt-increase and agreement-drop; improvements score zero |
| 9 | format seam rides in every comparison | FIXED: summary_native compressor (trace-format delivery) enables format-matched comparisons; wrapper_only remains the seam control |
| 10 | NO_ACTION conflates four failures | FIXED by parse_diagnosis categories |
| 11 | sampling-policy dependence | DOCUMENTED: all numbers conditional on temp 0.7 / top_p 1.0 / 8 samples; exp14 measured the top_p sensitivity |

All fixes unit-tested (suite extended, ALL CLEAR). Adoption note: new experiments
should report action_change_all + debiased excess + harm_score; existing results
stand as label-level TV with floors, as published.

## T4 final verdict (July 21): a clean NULL

Training completed properly (adapter saved, 86 gap-filtered on-policy pairs,
~30 optimizer steps). Powered task-split eval with CIs and paired test:
dpo D 0.73 [0.67,0.80] vs base 0.75 [0.68,0.81], acting 0.42 vs 0.46,
paired p = 0.345. No detectable effect in either direction. The E-B claim
("D works as an offline training signal") is NOT established at this scale;
E-A (selection) remains established. Options recorded in COAUTHOR.md:
1k-pair round, SFT-on-best arm, or publish selection-only.

## Still open (known, accepted, scheduled)

- Repo-level train/held-out split for E-B (before any writeup).
- On-policy trace replication.
- Multi-seed everything (only exp4 has 2 seeds so far).
- Second family (GLM) results pending.
- PYTHONUNBUFFERED in job scripts (cosmetic: live progress in .out files).
