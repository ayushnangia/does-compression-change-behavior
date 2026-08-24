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

## exp17 + final-batch dispositions (July 21)

- **exp17 (minimal behavioral core), N=19, coarse floor 0.13: the knee is
  below 2%.** keep_recent at 2% of old history (~330 tokens of 16k) still
  agrees 0.59 with the real action (25% kept: 0.68); the 12x compression
  from 25%->2% costs ~9 points. Extractive-2% beats abstractive summaries
  (0.30, exp8, same raw recent slice) by ~30 points. Skeleton (calls only)
  has the lowest coarse-D at R>=0.125. One model, coarse granularity.
- 32k/64k HF arms OOM'd (KV at 32k x 8 samples): parked pending the
  vLLM-scorer port of exp4/exp8.
- TB2-bf16 failed twice at vLLM TP=4 startup with zero log output even
  unbuffered: do not spend more blind queue cycles; needs an interactive
  salloc debug or the H100-cluster route.
- exp19 (timeout) and exp20 (NLL-cap OOM) refixed and resubmitted.

## exp20 verdict (July 21): the OOD bridge holds via containment

Spearman(3-gram containment vs original, coarse-D) = -0.55 across 6
conditions x 17 examples at 25% budget; condition means monotone
(keep_recent 1.00->0.13 D through wrapper_only 0.00->0.51). NLL is a poor
predictor (+0.11): fluency is not behavioral safety. Note-vs-trace summary
delivery identical for 9B at 4k (the format cliff is model/regime-specific).
Caveats: N=17, one rate, cluster-driven correlation. Predictive rule:
compact extractively; abstractive tokens cost behavior.

## exp19 verdict (July 23): sampled tool-level TV is noise-dominated; exact estimator confirms floor-referencing was load-bearing

On identical compressed-vs-full pairs, exact logprob TV over the next-tool
distribution averages 0.029 while the 8-sample estimate averages 0.275
(r=-0.32, meaningless at N=10). Reading: the true "which tool" shift under
compression is tiny; the sampled number at this granularity is ~90% sampling
floor. Our floor-referencing practice was therefore necessary and correct,
and headline effects (which run 0.5-0.8 vs floor ~0.33 and are carried by
the halting/acting channel) are unaffected. Caveat: the exact method
conditions on a tool call being made, so it cannot measure halting - the
channel exp4 identified as primary. The two estimators are complementary,
not interchangeable. Took 4 attempts (module-level method, 7.6GB logprob
spike, KV pre-allocation absorbing the freed headroom); final fix chunk=1024
+ util=0.80.

## exp21 verdict (July 23): canonical shorthand works if and only if it keeps the native wrapper

Group-proposed compressor: rewrite each tool call as GREP/READ/EDIT shorthand
(10.0x measured compression on tool-call tokens, 100% parse coverage).
N=19, floor 0.13, pre-registered predictions. P3 confirmed (p=0.013): identical
shorthand content scores 0.61 agreement inside <tool_calls> tags vs 0.42 bare -
19 points from format alone, our cleanest content-held-constant format test.
P2 directional (wrapped 0.61 vs raw skeleton 0.53 at 2% budget, p=0.34).
P1 wrong informatively: at moderate budgets raw skeleton beats canonical by
~5-9 pts, consistent with exp20's containment law (canonicalizing destroys
verbatim n-grams). Saturation result: the full 16k action history canonicalizes
to ~870 tokens holding 0.53-0.61 vs 0.68 for 3840 raw recent tokens.
Deployable rule: shorthand the action history, keep the native tool-call
wrapper. Caveats: N=19, one model (9B), one trace domain.

## Multiple-comparisons control (added 2026-07-24, hostile-review fix)

Benjamini-Hochberg at q=0.05 over the family of presented primary endpoints
(m=6): exp11 selection (p=0.0004), exp21 wrapper effect (p=0.013), exp4
freeze law (acting, p=0.016) PASS; exp6-coarse (p=0.074) and exp21
extreme-rate (p=0.34) fail and stay labeled suggestive; expB DPO
non-significance is the null claim itself. The three headline claims
survive family-wise correction; nothing presented as a finding fails it.

## exp4 requant verdict (July 24): freeze law SURVIVES, magnitude re-quoted

Under the fixed parser (5 formats) and deployment budget (10240): halts
0.31 (drop tool calls) vs 0.10 (control) - ratio 3.1x, absolute gap 21pts
(was 28). The old parser + 768 cap inflated halts EVERYWHERE (control
0.19 -> 0.10), confirming the bugs were real; the differential effect is
intact and observations remain free (0.09 vs 0.10). Label-level change is
at ceiling (floor 0.70) - halt/acting carries the finding, as before.
Usable N 15/24 under the stricter conditions; temp still 0.7 (X.1 open).
QUOTE THE NEW NUMBERS from here on: 0.31 vs 0.10.

## GLM requant verdict (July 24): the format cliff is 100% real

With the fixed parser (GLM native arg_key format now parsed) and both
top_p arms: native 0.75 acting, wrapper 0.00 acting, N=24 - identical to
the original numbers. Under wrapper delivery GLM narrates the task in
prose instead of acting; the collapse is behavioral, not parser blindness.
Threat 3.1 closed. Remaining on this claim: a temperature-1.0 arm (3.2).
Job note: GLM degenerate generations at deployment budget consumed the
12h wall; exp21 requant resubmitted separately; exp20b split per-rate at N=32.

## exp20b verdict (July 25): containment law replicates at every rate; magnitudes re-quoted

Three independent rates (5%/12.5%/50%), N=23 usable each, on-policy 16k,
fixed parser, deployment budget: Spearman(containment, coarse-D) = -0.38 /
-0.44 / -0.45. The original -0.55 was cluster-inflated as suspected; the
durable number is ~-0.4 at every rate. HONEST REVISION on NLL: +0.22..+0.30
(right direction, consistent) - "NLL predicts nothing" is retired; the claim
is now "containment is the consistently stronger predictor (~1.5-2x |rho|)".
Threat 2.1 closed.

## exp21 requant verdict (July 26): wrapper effect survives - AUDIT PHASE COMPLETE

Under fixed parser + deployment budget: wrapped 0.61 vs bare 0.43 agreement
at 2% budget (+18pts, p=0.023; was +19, p=0.013). Extreme-rate edge over raw
skeleton stays directional-only (p=0.278). Canonical saturation intact
(~481/869 tokens for the full history). With this, ALL FOUR headline claims
have survived hostile requantification: freeze law (re-quoted 0.31/0.10),
GLM cliff (reproduced exactly 0.00/0.75), containment law (re-quoted ~-0.4
x3 rates), wrapper effect (+18pts). None died; three were honestly re-quoted.

## Standard-runs verdict (July 27): X.1 closed for the core findings

exp4 at the FULL cited standard (certified parser, 10240 budget, temp 1.0):
freeze law HOLDS - halts 0.42 (drop tool calls) vs 0.21 (control), 2.0x;
observations still free (0.17 <= control). New at temp 1.0: drop_reasoning
halts 0.38 (was smallest effect at 0.7) - a temperature interaction worth a
powered look. exp21 at the standard: wrapper effect directionally intact at
every rate (+12pts at 2%, 0.45 vs 0.33) but p=0.121 at N=18 - temp 1.0 adds
noise everywhere (~10pts lower agreement across all conditions; 12h for 18
examples). STATUS: wrapper effect = significant at 0.7 (2x), directional at
1.0; the powered temp-1.0 run (N~64) is H200 material. Reporting standard:
quote requant numbers as primary, temp-1.0 as robustness, both on the table.

## Still open

Superseded (2026-07-31): the live open-items list moved to /STATUS.md
(single source of current state; this file stays append-only history).
Of the old list: on-policy replication DONE (16k on-policy requants,
Jul 24-26), GLM second-family results DONE (cliff requant Jul 24);
repo-level E-B split, multi-seed pass, PYTHONUNBUFFERED remain open -
tracked in STATUS.md.

## On-policy suite round 1 (2026-08-04): infrastructure complete, statistically PRELIMINARY — do not quote

First full 15-exp suite on on-policy data (examples_onpolicy.json, 64
full-context examples, certified parser, temp 1.0, chunked sampling).
All 15 completed; 4 infra failure modes fixed en route (STATUS ledger).
CRITICAL CAVEAT: the acting>=0.5 filter passes only ~25-30% of on-policy
examples (usable N=5-7 per exp; not a pure length effect — a 107k example
passed while 20k examples failed). NOTHING from this round is quotable;
powered N=64 reruns submitted (715808-15). Preliminary reads to verify:
- exp23 P1 NULL at N=6 (p=1.0): oneliner_tail 0.40 vs keep_recent 0.42
  @R=0.25. skeleton_tail (raw blocks + tail) directionally BEST at every
  rate (0.54/0.65/0.52 agreement) — consistent with the containment law
  (verbatim > canonical), would flip the exp22 arm-B design toward raw
  skeletons if it powers up.
- exp4 on-policy at N=5, floor 0.50 (weak regime): NO differential halting
  visible (drop_tool_calls none=0.30 vs control 0.28; drop_observations
  0.45 highest). If this survives the powered rerun it is a MAJOR revision
  to the freeze law's on-policy validity — treat as the single most
  important number in the 715808 harvest.
- Low full-context acting on long on-policy contexts is itself a
  finding-candidate (ties to C3/freeze): schedule a length-binned acting
  analysis from exp3's result.

## On-policy POWERED verdicts (2026-08-06, N=25 usable, jobs 715808-15): two headline laws do NOT replicate on-policy; the extractive-beats-abstractive family does

Regime: our own Terminus-2 trajectories (full real contexts 2k-209k,
compaction pressure restored), 9B measuring model, certified parser,
temp 1.0, chunked sampling, 25/64 examples pass the acting filter.

- **exp4 (freeze law): CLEAN NULL on-policy.** Halts 0.35 (drop tool
  calls) vs 0.28 (control), CIs overlap everywhere, paired p=0.78;
  observation-asymmetry also gone (0.33 vs 0.35). The 3.1x off-policy
  effect (and 2.0x @temp 1.0) does NOT transfer to this trace
  distribution. Freeze law must be re-scoped: established on off-policy
  fixed-16k <agent_trace> serializations; absent on on-policy full-context
  native-format traces at N=25.
- **exp21 (wrapper effect): NULL/REVERSED on-policy.** Wrapped 0.49 vs
  bare 0.55 @2% (p=0.36). The +18pts off-policy effect does not transfer.
  Canonical shorthand LOSES to raw skeleton at every rate on-policy.
- **exp23 (pre-registered): P1 NULL confirmed at power.** oneliner_tail
  0.61 vs keep_recent 0.68 @25% (p=0.43). skeleton_tail (raw blocks +
  verbatim tail) is directionally best at ALL rates (0.71/0.74/0.68 vs
  keep_recent 0.68/0.66/0.69; P3 p=0.11). Canonicalizing costs ~13pts vs
  raw skeleton+tail. Deployable-rule revision: keep it verbatim.
- **exp20 (containment): direction holds, magnitude drops.** Spearman
  -0.23 (was ~-0.4 off-policy); still ~2x NLL (-0.12). Summaries still
  cost acting (0.56-0.60 vs keep_recent 0.79).
- **exp8 (grounding): replicates.** keep_recent@25% agrees with logged
  actions 0.66 = full-context 0.65; summary 0.46 (-19pts). The
  extractive-free/abstractive-costly asymmetry is the robust cross-regime
  finding.
- Consistent picture across exp8/20/21/23: VERBATIM-EXTRACTIVE >>
  ABSTRACTIVE/CANONICAL on-policy; block-identity and wrapper-format
  effects were regime-specific.
- Pending: exp6p (running), exp14p (queued). Full-context acting collapse
  (25/64 pass) remains a finding-candidate needing length-binned analysis.

Paper consequence: Findings #1 (freeze) and #3-wrapper must be re-scoped
as off-policy/format-specific or dropped from headlines; Finding #6
(summaries worst) and the containment family are now the lead results,
with skeleton_tail as the deployable rule. OUTLINE updated.

## RETRACTION (2026-08-06): every TB2 Pass@1 = 0 on this infrastructure was a VERIFIER ARTIFACT, not a measurement

Root cause found by trajectory inspection: 82/89 TB2 tasks' tests/test.sh
curl-installs uv from astral.sh and uvx-downloads python3.13+pytest AT
VERIFY TIME. Compute nodes are air-gapped: curl times out (300s),
`uvx: command not found`, tests never execute, reward=0 unconditionally.
Verified across ALL runs: 0 verifier-clean trials out of 119 (bf16 Qwen
easy25+3 shards, GLM c64k). The agents themselves ran coherently (e.g.
118-turn trajectories with real analysis).

CONSEQUENCES:
- "TB2 Pass@1 = 0 is a robust negative" (July 21) is RETRACTED. The int4
  bf16 comparison built on it is unsupported. Honest statement: Pass@1
  has NEVER been validly measured on this cluster; model capability on
  TB2 is UNKNOWN here.
- The behavioral suite is UNAFFECTED (it never uses rewards); trajectories
  remain valid on-policy behavioral data - agents behaved identically,
  only scoring was broken.
- exp22's D->Pass@1 bridge is BLOCKED until verifiers run offline.

FIX (committed): bake.def.tpl now installs uv at bake time and pre-warms
the exact per-image union of uvx environments; %environment exports
PATH=/root/.local/bin and UV_OFFLINE=1 (cache-only instead of doomed
timeouts). NEW GATE before any Pass@1 is ever quoted: an offline verifier
smoke (apptainer --network none, run test.sh, require pytest to EXECUTE)
plus an oracle/solution run scoring ~1.0 on easy tasks. Old trials cannot
be re-verified (container filesystems were ephemeral); Pass@1 requires
fresh episodes on rebaked sifs.

Residual known limitation: tasks whose AGENT (not verifier) needs internet
remain unsolvable offline; the offline-solvable subset must be classified
before quoting rates.

## exp23 post-hoc contrast (2026-08-06, from saved raw arrays): skeleton_tail vs keep_recent is a TIE

Not pre-registered (P1/P3 were); computed from the powered run's raw
arrays before any slide/paper claim: +0.035/+0.085/-0.010 agreement at
R=0.25/0.5/0.75, pooled +0.037, paired p=0.39 (n=25). QUOTABLE CLAIM:
verbatim policies tie at the top; rewriting policies lose (canonical
-13pts, summary -20pts). "Skeleton+tail beats keep-recent" is NOT
established - do not claim a winner among verbatim policies.

## Verifier fix VALIDATED offline (2026-08-06): the Pass@1 axis is measurable for the first time

Round 1 of the fix failed its own smoke: apptainer fakeroot mounts the
HOST home over /root at runtime, shadowing everything baked there. Round 2
relocates uv to /usr/local/bin and caches to /opt (both PATH'd and
UV_OFFLINE'd in %environment). OFFLINE SMOKE GREEN on a rebaked sif with
network disabled: pytest executed a real assertion ("/app/out.html does
not exist", 1 failed in 0.70s) and wrote reward.txt. Reward 0 now means
UNSOLVED, not unmeasured. Gate for quoting Pass@1 hereafter: this offline
verifier smoke + an oracle-class run on rebaked sifs. easy25 rebaked
(25/25). **Aug-18 correction:** the full-89/eval auto-chain did not complete:
its process polling raced the rebake (88/89) and was not durable. The 64
non-easy25 images remain stale. This does not block the easy25 validity gate.

## Aug 8–18 final harvest and recovery decision

- **exp14 powered on-policy completed (job 715815), N=24:** at top_p=1,
  Qwen native acting=0.469 and wrapper=0.479; at top_p=0.9, 0.417 vs 0.490.
  There is no powered on-policy Qwen format cliff. The old GLM 0.00/0.75
  cliff is off-policy only and is cut from the main narrative.
- **exp6 timed out twice** (715809 and checkpointed 723922) at the same long
  example. The second job saved N=16 but restarted from zero: snapshots were
  added without resume semantics. Partial floor=0.453 [0.344,0.562], and no
  stable compressor ordering appears across rates/metrics. exp6 is retired
  from the ICLR critical path; exp8/17/23 answer the budget question with
  powered, directly grounded endpoints.
- **Oracle validity gate GREEN (2026-08-18):** with external networking
  blocked but localhost exempted for Selenium, the reference solution for
  `break-filter-js-from-html` passed real pytest and wrote reward=1. The gate
  is reproducible via `tb2/oracle_smoke.sh`. Valid Pass@1 is now measurable
  on rebaked images.
- **Selection-vs-length analysis (job 715810 indices, Aug 18):** 25/64 pass
  the acting filter, but failures are NOT concentrated at long contexts.
  Usable rates rise from 0.20 (<16k) to 0.40 (16–32k) and 0.50 (32–64k and
  64k+); usable median length=48.6k vs skipped=25.6k, mean-length permutation
  p=0.254. Retire the phrase “full-context acting collapse at long lengths.”
  The filter is a policy/state selection issue, not demonstrated length decay.
- **Program decision:** freeze duplicate exp3–21 proxy jobs and make exp22 the sole next objective.
  First establish a competent 35B easy25 baseline; then run keep-recent, raw
  skeleton+tail, stock summary, and no-compaction arms. If baseline success is
  at floor, switch model/scaffold rather than run more behavioral proxies.
- **exp22 implementation corrected before launch:** canonical one-liners were
  replaced by byte-exact command-bearing messages + verbatim tail (the exp23
  verdict); duplicate/missing cap logic was fixed; A/B/C now share the same
  post-third-compaction fallback.
- **Gate-1 launch caught a sampling mismatch before scoring (Aug 18):** vLLM
  imported Qwen's bundled generation defaults (`top_k=20, top_p=.95`) unless
  explicitly disabled, despite the declared 1.0/1.0 protocol. Job 803214 was
  cancelled before results. `--generation-config vllm` is now mandatory in
  TB2 and exp24 launchers and enforced by tests; the corrected baseline is the
  only run that may be interpreted.
- **Qwen3.8 preflight round 1 (803407): serving GREEN, test parser WRONG.**
  Qwen3.8-27B bf16 loaded in 51.1 GiB and served the full 77,824-token window
  on one H100; completions succeeded. The gate then incorrectly fed raw
  Terminus JSON responses to `behavior.parse_action`, which expects Harbor's
  post-action `<tool_calls>` trace serialization. Its all-None result is not a
  model verdict. Slurm `afterok` correctly cancelled baseline/data jobs
  803411/803412. Round 2 uses Harbor's authoritative
  `TerminusJSONPlainParser` and prints raw snippets before any decision.
- **Qwen3.8 preflight round 2 (803548): wrong interface, no model verdict.**
  Full-window serving again passed. The raw completion endpoint was fed an old
  Qwen3.5 serialized `<agent_trace>` prefix; Qwen3.8 closed that document and
  wrote retrospective prose. Harbor's parser correctly found zero commands,
  but live Terminus uses chat messages + Qwen3.8's native chat template, not
  that raw-prefix interface. Baseline/data were again safely cancelled by
  `afterok`. Round 3 removes the proxy entirely: one real Harbor/Terminus task
  must contain executed `tool_calls` in trajectory.json and real pytest output
  before easy25 is released.
- **Round-3 gate 805028 exposed Qwen3.8 timeout configuration.** Live Harbor
  integration itself works: 8 trajectory steps, 14 authority-parsed/executed
  tool calls, and real pytest/reward output. But the episode ended in
  `AgentTimeoutError` after LiteLLM's fixed 600s request timeout. The gate
  incorrectly passed because it checked commands+verifier but not
  `exception.txt`; that is now a hard failure. Dependent baseline 805029 ran
  for 12h and showed repeated identical 600s request timeouts, so it and data
  job 805030 were cancelled and are non-results. Root cause: Qwen3.8's native
  chat template defaults to `reasoning_effort=xhigh`; at ~15 tok/s a long
  response exceeds 600s. Fix: model-native `reasoning_effort: low` plus
  LiteLLM timeout=1800, while retaining the 10,240 output budget. The partial
  baseline directory is archived and never resumed/scored. The same audit
  found exp24's frozen-executor reward was also using raw `/v1/completions`;
  before any training, it was changed to native `/v1/chat/completions`, low
  reasoning effort, a post-compaction handoff message, and raw Terminus JSON
  parsing. Thus live evaluation and training reward now use the same interface.
- **exp24 plumbing pilot round 1 (809830) failed fast, no training:** the
  standalone data entrypoint imported `behavior.py` from the repo root without
  adding that root to `sys.path`; Slurm executes the script with only
  `experiments/` importable. Fixed after 10 seconds, before data or GPUs were
  used meaningfully. Regression test now launches the entrypoint from `/tmp`,
  so the test harness cannot mask this class of path bug by running at repo root.
- **exp24 pilot data round 2 (809850) GREEN:** immutable snapshot of five
  completed baseline trajectories produced 66 Qwen3.8-on-policy decision
  points from 3 tasks in 28 seconds. All three hash to train (0 validation),
  which is acceptable only for the 10-step plumbing pilot; the launcher now
  omits an empty eval file. This dataset is explicitly too small for evidence.
- **exp24 GRPO pilot round 1 (809859) failed before model load/training:** the
  launcher loaded the selector's Python 3.11 modules before invoking the
  Python-3.12 `ENV-vllm2`. On a clean `--export=NONE` node this omitted the
  Alliance opencv stack through which the vLLM environment receives
  `packaging` and other system wheels. It failed in 19 seconds with
  `ModuleNotFoundError: packaging`; no model or training state was touched.
  The executor now starts in an isolated, known-working Python 3.12 + arrow +
  opencv module subshell (matching the green live gate), while selector
  training retains Python 3.11.
- **exp24 GRPO pilot round 2 (813697) reached a healthy executor but failed
  before selector load/training:** Qwen3.8 loaded and `/health` returned 200;
  then clean-node ENV-compress2 lacked `typing_extensions`. The login-node
  import test had been falsely green because it inherited a richer module
  stack. Reproduction after `module purge` identified the complete selector
  stack: `StdEnv/2023`, Python 3.11, `python-build-bundle/2026a`,
  `scipy-stack/2026a`, arrow, gcc, and CUDA. The launcher now imports all
  training dependencies *before* starting the 27B server. A new one-H100 gate
  constructs Qwen3.5-4B + LoRA + the installed GRPOTrainer before another
  four-H100 pilot can be released.
- **selector gate round 1 (813707) failed fast; dependent pilot 813708 never
  ran:** on a pristine Slurm shell `StdEnv/2023` changes the hierarchical
  `MODULEPATH` and therefore cannot reliably be loaded in the same transaction
  as Python/build-bundle/scipy-stack. Because stderr had also been suppressed,
  the absent modules were visible only at the import assertion. StdEnv is now
  loaded separately, module errors are no longer hidden, and `module list` is
  recorded. This consumed 10 seconds on one H100, not a full-node pilot.
- **selector gate round 2 (813711) loaded every requested module but still
  failed fast:** the script replaced `PYTHONPATH` with the repository root,
  deleting Alliance's `sitecustomize` path. Without that hook Python does not
  expand `EBPYTHONPREFIXES`, so module-provided `typing_extensions` remained
  invisible despite appearing in `module list`. Both exp24 launchers now
  prepend the repo (`$ROOT:${PYTHONPATH:-}`) instead of replacing the cluster
  path. This again consumed only 10 seconds on one H100.
- **selector gate round 3 (813716) reached model+LoRA/TRL construction:** the
  full Qwen3.5-4B weights loaded, then installed TRL 0.29 rejected the stateful
  reward callable because it assumes every reward function has `__name__`.
  `ExecutorReward` now exposes the stable name `executor_behavior_reward`.
  This was the first gate to clear dependencies, data loading, lineage, and
  model loading; it stopped after 35 seconds on one H100.
- **selector gate round 4 (813721) GREEN:** complete clean-node dependencies,
  66-row pilot data and lineage guard, full Qwen3.5-4B load, LoRA attachment,
  and installed TRL GRPOTrainer construction all completed in 33 seconds on
  one H100. This authorized only the 10-step integration pilot 813724, not a
  powered result.
- **integration pilot 813724 cancelled after step 1/10 as invalid:** both
  models loaded, executor `/health` was green, TRL generated/scored a group,
  and backward plumbing ran. But all 8 logged selector completions were trace
  fragments rather than JSON, hence reward=-1 for every candidate, group
  variance=0, loss=0, grad_norm=0. No checkpoint was saved or interpreted.
  Root cause: prepared `prompt` was a plain string, so TRL performed raw LM
  continuation rather than applying Qwen3.5's native chat template; the long
  block manifest also separated generation from the initial JSON instruction.
  Fix: conversational user-message prompts, repeated JSON contract at the
  generation boundary, and reward normalization for TRL's conversational
  completion structure. The one-H100 gate now requires >=1/4 parseable JSON
  generations before another full-node allocation.
- **native-chat selector gate 828054 still refused 0/4 JSON outputs:** unlike
  813724, all candidates correctly understood and analyzed the selection task,
  proving chat formatting was fixed. But Qwen3.5's template defaults to an
  open `<think>` block, so four 64-token completions ended during prose before
  producing JSON. Use Qwen3.5's native `enable_thinking=False` template option
  in both TRL rollout and preflight. This preserves the intentionally short
  structured-action budget and does not weaken the parser or reward gate.
- **native no-thinking selector gate 828065 GREEN:** 4/4 sampled completions
  parsed under the strict selector JSON parser, with diverse keep sets (short,
  all-block, and intermediate selections). Qwen3.5-4B + LoRA + installed TRL
  construction also remained green. This releases only dependent 10-step
  integration pilot 828066; budget-overflow candidates still receive the hard
  penalty during reward execution.
