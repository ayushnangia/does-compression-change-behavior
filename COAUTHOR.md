# The compression-behavior project: complete technical briefing

This document is self-contained. It assumes you know what an LLM is and
nothing else about this project. Every number has a JSON in
`experiments/results/`, every claim has a confound entry in `AUDIT.md`,
and section 12 tells you which results to trust. Written to be checked,
not believed.

---

## 1. The problem, from the beginning

An LLM agent solves long tasks by looping: read the context, emit a tool
call (run a command, edit a file), receive the output, repeat. Everything
so far (its actions, the outputs, its reasoning) accumulates in the context
window. Real tasks outgrow the window mid-task.

Every production system (Claude Code, OpenHands, Terminus) handles this the
same way, called compaction: replace the old history with an LLM-written
summary, keep the most recent turns raw, continue. Compaction quality is
checked with information metrics: perplexity, QA probes, "are the facts
still there?"

The unasked question: after compaction, is the agent's next action still
the action it would have taken? Keeping facts is not the same as keeping
the policy. That gap is the entire project.

## 2. The framing

Lossy compression is a rate-distortion problem: you trade rate (how much
you keep) against distortion (what you lose), under a distortion measure
that you choose. Classical choices measure reconstruction error. We choose:

    distortion D = how much the agent's next-action distribution changes
                   when the context is compressed.

This makes compression task-agnostic and self-supervised: any trajectory is
training signal, no labels needed. The information-bottleneck way to say
it: keep the bits that predict behavior, discard the rest.

## 3. How D is measured, exactly

### 3.1 Decision points
A decision point is a spot in a real agent trace immediately before a tool
call. The tokens up to that point are the context. Every compressed variant
keeps the most recent slice raw (512 tokens at 4k contexts, 1024-2048 at
16k+), mirroring real systems, and replaces only the older history.

### 3.2 Sampling behavior
From a context, generate 8 continuations (temperature 0.7, top_p 1.0, up to
768 new tokens). Parse each into an action label:
- a tool call becomes `tool_name::first-60-chars-of-arguments`,
  for example `execute_bash::{"command": "grep -n Parse json_test.go"}`;
- no tool call (prose, runaway `<think>`, gibberish) becomes `NO_ACTION`.

Arguments are included because agent traces are dominated by one tool
(`execute_bash`), so name-only labels would hide most behavioral change.
The parser (`behavior.py:parse_action`) handles every syntax we have met:
`<tool_calls>` plural (trace format), `<tool_call>` singular (Qwen3 chat),
Python-repr dicts with single quotes, Terminus's `function_name` field,
`<think>` blocks (stripped; a call inside thinking is not an executed
action), and arbitrary literals in arguments (sets crashed json.dumps once
and killed two jobs; serialization is now total). `parse_diagnosis`
classifies every non-action as prose / toolish_unparsed / think_runaway /
empty; a nonzero toolish_unparsed rate means investigate the parser, not
the model.

### 3.3 The distance
Sample 8 actions with the full context, 8 with the compressed one. Each set
is an empirical distribution over labels. D is the total variation distance:

    D = 1/2 * sum over labels a of | p_full(a) - p_comp(a) |

Worked example. Full: {grep x4, edit x3, NO_ACTION x1}. Compressed:
{grep x2, NO_ACTION x5, run_tests x1}.

    | label     | p_full | p_comp | diff |
    | grep      | .500   | .250   | .250 |
    | edit      | .375   | 0      | .375 |
    | NO_ACTION | .125   | .625   | .500 |
    | run_tests | 0      | .125   | .125 |
    D = 1/2 * 1.25 = 0.625

Intuition: out of 8 samples, how many would you have to swap in the second
list to make its tally match the first (5 of 8 here). 0 = identical
behavior mix, 1 = disjoint.

### 3.4 The noise floor
Sampling is stochastic: the same context sampled twice with different seeds
already gives D between 0.25 and 0.62 depending on regime. Every experiment
measures this floor first and reads effects against it, with paired tests
(all conditions share examples and seeds, so differences are
within-example).

### 3.5 Granularities and estimator upgrades (added after self-critique)
- label level (name + args, saturates near 1.0 in noisy regimes),
- verb level (tool + first command token: `bash ls` vs `bash git`),
- tool level (name only, lowest ceiling).
  Standard: report all three; a conclusion that holds at one granularity
  only is an artifact.
- `debiased_change`: plug-in TV from 8 samples is inflated, and the
  inflation depends on the distributions' entropy, so a shared floor is
  only approximately fair. Fix: pool the 16 samples, permute the
  assignment, subtract the per-comparison null mean. Gives a debiased
  effect size and an exact p.
- `harm_score`: plain D penalizes beneficial divergence (deleting reasoning
  IMPROVED acting in exp4, and D scored that as damage). harm_score counts
  only halt-increase and lost agreement with the logged action.
- exact logprob method (`vllm_scorer.exact_tool_distribution`): at
  tool-name granularity the next-action distribution can be computed
  exactly by scoring each candidate tool continuation. No sampling, no
  floor. Validation vs the sampled estimate is exp19 (in flight).

### 3.6 Companion metrics
D alone hides HOW behavior broke, so it is always reported with:
- acting rate: fraction of samples that are a real tool call;
- agreement: fraction matching the action the ORIGINAL agent actually took
  at that decision point (mined from the trace; the external answer key);
- action entropy: is the uncertainty structure preserved, not just the mode;
- kind profile: halt / lookup / commit decomposition;
- ngram_containment (input side, from Zhang & Khattab 2026): fraction of
  the compressed context's word 3-grams present in the original. 1.0 =
  purely extractive (truncation), low = abstractive (summaries).

## 4. Data

### 4.1 Off-policy traces
`nvidia/Open-SWE-Traces` (real software-agent runs, written by a 122B
model). Serialized into a tagged text format (`<turn role=...>`,
`<content>`, `<tool_calls>`), cut at decision points, tokenized into
windows (4096 tokens, recent 512). Caveat: written by a different model
than the one measured; within-experiment contrasts are still valid, and the
key results were later replicated on-policy.

### 4.2 On-policy trajectories
Our Terminal-Bench runs produced 129 trajectories generated by our own
models in live containers (median ~21k tokens, max 228k), including 132
genuine window-exhaustion compaction events with the summaries the
production agent actually wrote. `prefetch_onpolicy.py` converts them into
decision-point examples at 4k/16k/32k/64k, each carrying the logged next
action and up to 3 future segments (for teacher-forced multi-step
experiments).

### 4.3 Token composition (measured, sums to 100)
tool calls 25%, observations (tool outputs) 36%, reasoning prose 3%,
structural markup and user text 35%.

## 5. Experiments, one paragraph each

Numbers cited are means; CIs and raw arrays are in the result JSONs.

- **exp1 (analytic).** CompactionRL discounts early summaries by
  (gamma*lambda)^(tokens after the segment). With their hyperparameters the
  first summary in a 4-compaction chain trains at 13.5% of the last one's
  signal, x0.51 per step, scale invariant. Arithmetic from their equations.
- **exp2 (Monte Carlo).** If compaction segments are treated as GRPO group
  samples, a 4-segment rollout carries 3.2x the gradient mass of an
  unsplit one for the same outcome, and success advantages inflate +0.37
  when compaction correlates with difficulty.
- **exp3.** Target stability: the full-context reference itself drifts with
  context length; a padding control (same information, restored length)
  drifts almost as much as truncation (0.56 vs 0.60), so the sensitivity is
  positional, not informational. Consequence: training targets should be
  marginalized over lengths (adopted in exp16).
- **exp4 (the block law; strongest result).** Delete each content type at a
  matched 50% budget: tool calls gone -> acting 0.81 to 0.53, halting
  triples; observations gone (36% of tokens) -> indistinguishable from
  random deletion (D 0.50 vs 0.51); reasoning -> mild. Replicated on
  GLM-4.7-Flash, five runs, two seeds, and on-policy (halting axis). v1 had
  a mislabeled regex (its "observation" eraser deleted nothing), caught
  because the condition exactly matched control, fixed role-aware, rerun.
- **exp5.** Same summary, four wrapper formats: acting moves 12 points on
  format alone. Motivation for exp14, which is the citable version.
- **exp6.** The D(R) curve at 12.5-75% budgets: LLM summaries are the worst
  curve at every budget (0.68 vs keep-recent 0.49 at 25%, floor 0.36).
  Off-policy 4k scope. The on-policy version at label granularity is
  ceiling-limited (floor 0.61, uninformative); at tool granularity
  (exp6-coarse) the ordering CHANGES: block-aware becomes best
  (0.23 [0.13,0.34] vs keep-recent 0.47 [0.35,0.62], p=0.074, N=13,
  suggestive only, N=48 rerun needed).
- **exp7.** Iterated summarization (summary of summary): the first
  compaction does nearly all the label-level damage (2.3x floor
  immediately); deeper chains mainly add halting (0.53 -> 0.46). The GLM
  arm is discarded (it measured exp14's format collapse, not chain depth).
- **exp8 (grounding; the validity anchor).** Agreement with the logged real
  action: full 0.73, truncation 0.67, summary 0.30. Replicated at 16k and
  on-policy (0.74 / 0.67 / 0.30). High D = low agreement with reality.
- **exp9.** Summary-policy race at equal budget: the prompt derived from
  exp4 ("copy tool calls and paths verbatim, compress the rest") beats
  naive (acting 0.80 vs 0.51, agreement 0.55 vs 0.33) and Claude-Code /
  OpenHands style prompts (0.61-0.67 / 0.34-0.38). Measurement -> design.
- **exp10.** Teacher-force the real continuation after one compaction:
  divergence 0.86 -> 0.47 and acting 0.43 -> 0.90 within ~2 steps. Damage
  is transient UNDER THE REAL CONTINUATION (teacher-forced = upper bound on
  healing; the free-running variant is designed, not run).
- **exp11 (E-A, selection).** Best-of-8 summaries by argmin D
  (fresh-rescored to kill winner's curse): agreement 0.55 vs 0.38 random,
  paired p=0.0004. Replicated at 16k on 177 examples: D 0.67 [0.64,0.69]
  selected vs 0.72 [0.69,0.74] random (significant, small).
- **exp12.** Portability baseline: base-model summaries transfer across
  executors (9B<->4B, D 0.79-0.86 relative to each executor's floor). The
  pre-registered co-adaptation detector for any trained compressor.
- **exp13.** The deletion-manifest idea (inject "what was deleted and how
  to retrieve it"): +4 acting, +8 lookup share, no gain combined with a
  summary. BUT the recovery harness already advertises retrieval, masking
  the tested effect: an uninterpretable null, harness redesign required.
- **exp14 (format phenotypes).** Same summary, {trace wrapper vs native
  chat template} x {top_p 1.0, 0.9} x three models: GLM-4.7-Flash collapses
  to incoherence on the wrapper (acting 0.00, both top_p) and is fully
  rescued by its native format (0.75); Qwen3.5-35B-A3B (its MoE twin) has
  the OPPOSITE preference (0.79 wrapper vs 0.46 native); dense Qwen3.5-27B
  is robust to both. No universal compaction format exists.
- **exp15.** Score production compactions at the 132 real window-exhaustion
  events. Three attempts: two context-construction artifacts (post
  segments got a fresh document header; fixed by single-serialization
  slicing), and the final run is dominated by the format effect (wrapper
  conditions 0.00 acting), so the production-vs-designed comparison never
  happened. Only quotable: truncation agrees 0.74 with the real action at
  real boundaries (N=19). The format-matched instrument (summary_native,
  trace-format delivery) now exists for the redo.
- **exp16 (T4 phase 1).** Scaled on-policy pair generation via the vLLM
  scorer: 177 usable examples at 16k, marginalized two-length reference
  (floor drops to 0.253), 86 pairs above the 0.25 noise gap, task-level
  train/held-out split, trainee-written candidates.
- **E-B (T4, training).** Ten attempts. Eight were infrastructure (trl API
  rename, OOM, three flag incompatibilities discovered by test, allocator
  fragmentation, a torch-version env-var rename, off-policy pairs). The
  ninth trained nothing (documented). The tenth completed cleanly and is a
  NULL: dpo D 0.73 [0.67,0.80] vs base 0.75 [0.68,0.81], p=0.345. D as a
  training signal is NOT established at ~100 pairs.
- **exp17 (in flight).** Minimal behavioral core: rates down to 2% kept,
  including an action-skeleton condition (context = only the tool calls).
- **exp18 arms (in flight).** exp8/exp4 at 32k and 64k on window-filling
  on-policy examples.
- **exp19 (in flight).** Exact logprob tool distribution vs the sampled
  estimate: correlation and noise elimination.
- **exp20 (in flight).** The OOD bridge: does input-side
  out-of-distribution-ness (containment, NLL) predict D? Includes
  summary_note vs summary_trace (same content, different format).
- **Terminal-Bench 2.0 grounding.** Full offline pipeline on an air-gapped
  cluster (89 task containers rebuilt for Apptainer, models via vLLM,
  Terminus-2 agent). All quantized models scored 0% Pass@1 (0/53, 0/69
  default budgets; 0/21 easiest-25 at 4x timeout; mostly genuine timeouts).
  Valid negative, no discriminative power. The bf16 35B run (32k window,
  forcing real compaction) is in flight; it is also the source of on-policy
  trajectories from a competent model.

## 6. The findings as laws (with their evidence tier)

1. Behavioral value concentrates in the action history, not the
   informational bulk (STRONG: exp4, cross-family, on-policy).
2. Information-optimized summaries are the worst compressor at every budget
   on off-policy 4k data (MODERATE: exp6); on-policy the measured
   block-aware design wins at tool granularity (WEAK until N=48: p=0.074).
3. Delivery format is first-order and model-specific, from 12-point
   degradation to total collapse, with no universal format (STRONG: exp14).
4. Compaction damage is front-loaded (exp7) while RL credit is back-loaded
   (exp1), but the damage is transient under the real continuation (exp10),
   so the practical conclusion is to grade summaries densely at their
   boundary rather than re-weight temporal credit.
5. D is externally valid (STRONG: exp8) and works for selection (STRONG:
   exp11); it does not yet demonstrably work for training (NULL: T4).
6. Geometrically: the model is not smoothly dependent on its context.
   Content directions are nearly flat; format directions are cliffs.

## 7. Self-corrections (read these to calibrate trust in the rest)

1. exp4 v1 mislabeled: "delete observations" deleted nothing; caught
   because it matched control exactly.
2. exp11 winner's curse: selected summaries were first scored on their
   selection samples; fixed with fresh-seed rescoring before results.
3. exp15 x2: separately-serialized segments gave wrapper conditions a fresh
   document header (model behaved as if the task was starting).
4. The round-1 "DPO made it worse" was FALSE: training-log forensics showed
   preference accuracy below chance and margins ~0; it had never trained
   (9B-written pairs used to train the 4B = off-policy preferences).
5. The healing result (exp10) genuinely weakened our own credit-assignment
   attack; reported as such.
6. exp13's null is confounded by its own harness; reported as
   uninterpretable rather than negative.
7. The gitignore pattern `results/` silently excluded experiments/results
   from git for the project's whole life; found during the filesystem
   incident recovery.

## 8. Statistics methodology

Floors before effects, always. Paired designs everywhere (conditions share
examples and seeds); paired permutation tests on pre-registered primary
endpoints (one per experiment); bootstrap CIs on means; raw per-example
arrays saved in every result JSON (post-audit); N=13-48 per cell, mostly
single seed (exp4 has two); all inference labeled exploratory-grade.
Estimator upgrades (debiased TV, exact logprobs) exist in code; headline
numbers predate them and should be rerun before the paper.

## 9. Infrastructure (what you inherit, all tested)

- Cluster: Narval (Alliance Canada), A100-40GB nodes, no internet on
  compute nodes. Everything runs offline: models pre-cached to $SCRATCH/hf,
  examples prefetched to JSON, Terminal-Bench task images rebuilt as
  Apptainer .sif with server dependencies baked in (compute nodes cannot
  pip install).
- Repo: $SCRATCH/dccb (GitHub is the real backup; scratch purges at 60
  days). Venvs: $SCRATCH/ENV-compress2 (torch 2.9.1 stack + trl/peft/fla),
  ENV-vllm2 (vllm 0.25), ENV-harbor2 (harbor 0.20). The home filesystem had
  a corruption incident on 2026-07-17 (~1000 truncated files across the old
  venvs); do not build anything under $HOME.
- Scoring at scale: `experiments/vllm_scorer.py`, batched in-process vLLM,
  6.2x the HF path, distributionally equivalence-tested (excess -0.016 to
  +0.031, CI spans 0), plus exact logprob tool distributions.
- Test suite: `tests/run_tests.py` (70+ checks incl. parser totality,
  statistics calibration, cross-script invariants; gate every submission
  with it) and `tests/test_dpo_pipeline.py` (CPU end-to-end DPO on a tiny
  model; caught one of the ten failures before it cost a queue slot).
- Job hygiene learned the hard way: pre-validate every external interface
  by CONSUMING it (not inspecting it) on the login node; both
  PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF (torch renamed it);
  PYTHONUNBUFFERED=1; sbatch copies scripts at submission; job outputs and
  logs go to scratch.

## 10. Literature position (RELATED.md has the full audit)

- CompactionRL (arXiv:2607.05378): the RL-trains-compaction work we
  analyze; our exp1/exp2/exp5/exp14 ground its credit scheme, GRPO
  mismatch, and transfer failure.
- LLMLingua-2 (2403.12968): closest compression prior, information-centric;
  we change the objective to behavior.
- StreamingLLM (2309.17453): recency dominance at the KV level; convergent.
- Agent-memory survey (2606.24775): independently documents the evaluation
  gap we fill (memory judged only by end-to-end scores).
- Metric lineage: TV between action distributions = TRPO's policy distance;
  logged-action agreement = imitation learning's action matching; adopted
  upgrades from UQ literature (semantic clustering per semantic-entropy /
  Kernel Language Entropy; MMD two-sample tests) are planned, not built.
- Zhang & Khattab 2026 (harness generalization blog): their input-side
  trajectory metrics + our output-side D are two halves of one
  delta-to-epsilon picture; our anisotropy finding refines their smoothness
  assumption; exp20 is the quantitative bridge.
- Caveat: all keyword-search depth; the citation-graph walk is still owed
  before any external novelty claim.

## 11. What will NOT stand (do not present as findings)

exp6-coarse's on-policy flip (p=0.074, N=13); exp13's manifest null
(confounded harness); exp15's production comparison (format-dominated, the
comparison never happened; only the truncation-0.74 number survives, on a
reconstructed reference); exp7's GLM arm (measured the format effect);
anything at label granularity in floor>0.5 regimes; all in-flight
experiments until their JSONs exist.

## 12. Priorities (costs in queue-days)

1. Harvest in-flight: exp17, 32k/64k arms, exp19, exp20, bf16 TB2 (0).
2. Power the two deciders: exp6-coarse at N=48; exp15 redo with
   summary_native format-matching (1 each).
3. E-B round 3 decision: SFT-on-best arm first (cheap), then either a
   1k-pair round via the vLLM pipeline (2-3) or publish selection-only.
4. exp10-free (free-running propagation) to complete the healing claim (1).
5. Rerun the headline table under the new estimator standard (1).
6. Multi-seed pass on exp4/exp6/exp9 (2).
7. Second trace domain for the block law, the highest-value generalization
   check (2-3 including plumbing).
8. Citation-graph walk, then the paper. RQ.md + this document's sections 5
   and 6 are the results-section outline.

## 13. File map

    behavior.py           sampling, parsing (total), diagnosis
    metrics.py            TV (3 granularities), debiased, harm, entropy,
                          containment, floors helpers
    stats.py              bootstrap CI, paired permutation
    data.py               trace serialization, Example schema, file I/O
    prefetch*.py          off-policy and on-policy example builders
    compressors.py        keep_recent/summary/summary_native/pointer/
                          hallucinator/paraphrase
    scaffold.py           the recovery test (lookups served from history)
    experiments/          exp*.py + job scripts + vllm_scorer.py
    experiments/results/  every result JSON (raw arrays included)
    tests/                the gate; run before every submission
    slides.html           the group deck
    AUDIT.md              claim-by-claim confound ledger
    RQ.md / RELATED.md / PLAN.md
