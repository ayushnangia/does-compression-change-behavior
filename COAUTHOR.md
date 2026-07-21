# Co-author briefing: the compression-behavior project, unvarnished

Everything below is written to be checked, not believed. Every number has a
result JSON in `experiments/results/`, every claim has a confound entry in
`AUDIT.md`, and the weak results are labeled as weak. Slides for the group:
`slides.html`. Formal question: `RQ.md`. Novelty audit: `RELATED.md`.

## The project in one paragraph

Agents compact their context when it fills; everyone checks whether the
facts survive, nobody checks whether the behavior survives. We built a
behavioral distortion measure D (total variation between sampled next-action
distributions, floor-referenced, externally validated against the agent's
real logged actions), used it to find design laws (what content matters,
what formats break models), and tested whether D can select and train better
compressors. Selection works. Training is a null so far. Task-level
grounding is still missing. The measurement infrastructure is solid; the
statistical power on several findings is not.

## Verdicts on the core ladder

| claim | verdict | evidence |
|---|---|---|
| D is measurable above noise | YES | floors measured per experiment (0.25-0.62 depending on regime) |
| D is externally valid | YES | tracks agreement with real logged actions (full 0.73 / truncation 0.67 / summary 0.30; replicated 4k, 16k, on-policy) |
| D selects better summaries (E-A) | YES | best-of-8: agreement 0.55 vs 0.38, p=0.0004; replicated at 16k on 177 examples with non-overlapping CIs (0.67 vs 0.72), though the 16k effect is small in absolute terms |
| D trains better summaries (E-B) | **NULL** | after 10 attempts (8 infra failures documented), a clean run: 86 on-policy pairs, ~30 steps, task-split powered eval: dpo 0.73 vs base 0.75, p=0.345. No effect either way. The training claim is NOT established |
| D predicts task success (E-D) | **UNTESTED** | all models scored 0% on Terminal-Bench (robust: 3 runs, 4x timeouts); bf16 35B run in flight is the last shot at this scale |

## Strength tiers, per experiment (the review you asked for)

### Will stand (strong)
- **exp4 v2, the block law.** Deleting tool calls triples halting; deleting
  observations (36% of tokens) is indistinguishable from random deletion.
  Replicated: 2 model families, 5 runs, 2 seeds, on-policy (halting axis).
  Survived its own bug (v1 mislabeled, caught via anomaly, rerun). This is
  the paper's anchor result.
- **exp8, grounding.** External answer key, three regimes, near-identical
  numbers. The reason anyone should trust D at all.
- **exp14, format phenotypes.** GLM: 0.00 wrapper -> 0.75 native (rescue).
  Qwen MoE: inverted preference. Qwen dense: robust. N=6/model but the
  effects are night-and-day, sampling-controlled, and the GLM collapse was
  adjudicated against parser failure with raw dumps. Qualitatively strong.
- **exp11 E-A, selection.** p=0.0004, winner's-curse-guarded, 16k
  replication at scale. Statistically the cleanest positive.
- **exp1/exp2, the analytics.** Exact computation and simulation. They are
  arithmetic; they stand by construction. Their *relevance* is arguable,
  their correctness is not.

### Moderate (usable with stated scope)
- **exp6, off-policy D(R).** Summaries worst at every budget. N=14, one
  seed, one trace source. Direction is consistent everywhere we looked, but
  a referee will ask for more seeds and a second domain and be right.
- **exp9, policy audit.** Large gaps (acting 0.80 vs 0.51), N=19. The
  block-aware-prompt-wins story is real but single-model, single-budget.
- **exp3, target stability + padding control.** Good design, N=15. The
  positional-sensitivity finding motivates marginalized targets; fine as a
  methods point.
- **exp10, healing.** Real and important BUT teacher-forced only = upper
  bound on recovery. The free-running variant was designed and never run.
  Quote as "damage is transient under the real continuation," never as
  "damage heals" unqualified.
- **exp16 phase 1.** 177 examples, marginalized targets, the machine works.
  The select-by-D gap at 16k (0.67 vs 0.72) is significant but small; do not
  oversell.

### Will NOT stand as currently evidenced (weak, do not present as findings)
- **exp6-coarse, "block-aware wins on-policy."** N=13, p=0.074/0.095.
  Suggestive, on the slides with hedges, but one skeptical question kills
  it. Needs the N=48 rerun before it is a claim.
- **exp13, the manifest null.** N=19 (12 scaffold), and the recovery harness
  itself advertises retrieval, masking exactly the effect being tested. A
  null with a known confound is uninterpretable, not a negative finding.
  Status should be "untested, harness redesign required," which is what the
  menu-free retest is.
- **exp15, production compactions.** Three attempts: two construction
  artifacts, and the final run is dominated by the format effect (wrapper
  conditions at 0.00 acting), so the production-vs-designed comparison never
  actually happened. The single quotable number (keep_recent agrees 0.74 at
  real boundaries, N=19) sits on a reconstructed reference of untested
  validity. Treat the whole experiment as a designed-but-not-yet-executed
  study whose instrument (summary_native, format-matched delivery) now
  exists.
- **exp7's GLM arm.** Superseded by exp14's finding that the wrapper format
  collapses GLM: the chain numbers measure the format effect, not chain
  depth. Discard.
- **exp5 in isolation.** Fine as the motivation for exp14, which is the
  citable version of the format claim.
- **On-policy exp6 at label granularity.** Floor 0.61: structurally
  uninformative regime. Already documented; never quote.
- **Anything from exp17/18/19/20 and the bf16 TB2 run.** In flight, zero
  results yet.

## Cross-cutting weaknesses (a referee's opening paragraph)

1. One trace ecosystem (SWE agents / Terminus), one primary measuring model
   (Qwen3.5-9B). Every "law" could be ecosystem-specific.
2. Ns of 13-24 for most cells, single seed for most experiments. Only exp4
   has meaningful replication depth.
3. The published numbers use the plug-in TV estimator; the debiased
   estimator, verb granularity, harm score, and exact logprob method exist
   in code but no headline experiment has been rerun under them.
4. No consequence grounding: D counts a trivial divergence the same as a
   task-ending one. bf16 TB2 is the only path at our scale.
5. The literature grounding is keyword-search depth; the citation-graph
   walk (LLMLingua-2, semantic entropy, agent-memory survey forward
   references) is still owed before any external novelty claim.
6. exp15/exp13 style format confounds mean any experiment mixing content
   and delivery format needs the summary_native control from now on.

## What must be done, in order (with rough costs)

1. **Land the in-flight jobs** (exp17 minimal-core, 32k/64k arms, exp19
   exact-vs-sampled, exp20 OOD bridge, bf16 TB2). Zero new effort.
2. **Power the two weak-but-promising results**: exp6-coarse at N=48 (one
   8h job) and exp15 with summary_native format-matching (one job; the
   compressor exists). These decide whether "block-aware wins on-policy"
   and the production-compaction study become claims or footnotes.
3. **E-B round 3 decision**: either scale to ~1k pairs (needs the vLLM
   scorer pipeline we built; ~2-3 days of queue) or accept
   "selection-yes, training-null-at-100-pairs" as the paper's honest
   position. Recommend attempting SFT-on-best as the cheap third arm first
   (one job): if SFT also nulls, the story is "the signal selects but is
   too weak/noisy to train against at this scale," which is coherent.
4. **exp10-free** (free-running propagation): completes the healing claim.
   One job.
5. **Rerun the headline table under the new metrics standard** (debiased,
   three granularities, harm score) so the paper's numbers use the
   defensible estimator. One job.
6. **Multi-seed pass** on exp4/exp6/exp9 (seeds 1,2). Two jobs.
7. **Second domain** (non-SWE traces: web-agent or tool-use dataset) for
   the block law only. This is the single highest-value generalization
   check. 2-3 days including prefetch plumbing.
8. **Citation-graph walk + paper skeleton.** RQ.md and RELATED.md are the
   inputs; the claims table in this file is the outline of the results
   section.

## Infrastructure you inherit (all working, all tested)

- Measurement: `behavior.py` (parser is total over model outputs, with
  parse_diagnosis), `metrics.py` (TV at 3 granularities, debiased
  estimator with exact p, harm score, n-gram containment),
  `stats.py` (bootstrap CIs, paired permutation).
- Scale: `experiments/vllm_scorer.py` (batched, 6.2x, equivalence-tested,
  plus exact logprob tool distributions).
- Data: on-policy prefetch from TB2 trajectories at 4k/16k/32k/64k, with
  logged actions and future segments; 132 real compaction events located.
- Eval: offline Terminal-Bench 2.0 pipeline (89 Apptainer images) on an
  air-gapped cluster; bf16 35B serving config.
- Hygiene: 70+ check test suite (gates every submission), AUDIT.md ledger,
  10 documented DPO failure modes so nobody repeats them.
- Ops notes: home filesystem had a corruption incident (2026-07-17);
  everything lives on $SCRATCH now ($SCRATCH/dccb, $SCRATCH/ENV-*2);
  fairshare on both accounts is warm; scratch purges at 60 days, so push
  often (the repo remote is the real backup).

## The one-sentence version for the paper

Information preservation and behavior preservation are different objectives:
we can measure the difference above noise, ground it in real actions, and
use it to select compressors; whether it can train them, and whether it
predicts task outcomes, are the two open questions the next month decides.
