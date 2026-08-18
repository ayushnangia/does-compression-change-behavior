# ICLR outline — narrative truth

Working title: **Does Compression Change Behavior? On-Policy Evaluation of
Context Compaction for Coding Agents**

Thesis: context compaction should be evaluated by the behavior and task
outcomes it induces, not by textual information retention. On our agent’s own
trajectories, verbatim-extractive policies preserve next-action behavior while
rewriting loses it; several apparent “laws” discovered off-policy disappear.

Data policy: paper claims use on-policy trajectories only. Off-policy results
appear only in the regime-dependence experiment. Never claim skeleton+tail
beats keep-recent: they tie (pooled p=0.39).

## Paper structure

### 1. Introduction

- Production agents summarize when context fills, but evaluate summaries as
  text rather than interventions on a policy.
- We define behavioral preservation and ask three questions:
  1. Which compression operations preserve next actions at fixed rate?
  2. Do conclusions transfer from public/off-policy traces to the evaluated
     agent’s own trajectories?
  3. Do behavioral differences predict task success in live episodes?
- Lead result: at 25%, keep-recent agrees with logged actions as often as full
  context (0.66 vs 0.65), while summaries fall to 0.46; rewriting a raw action
  skeleton loses about 13 points relative to keeping it verbatim.

### 2. Related work

Context compression/summarization, KV/cache eviction, agent memory, behavioral
distillation, CompactionRL, and rate-distortion. exp1/2 appear only as brief
motivation. Complete citation graph from `paper/LITERATURE.md` during writing;
it does not block exp22.

### 3. Method

- On-policy Terminal-Bench/Terminus trajectories and decision-point extraction.
- Full-context reference, matched token budgets, shared examples/seeds.
- Acting rate, logged-action agreement, label/tool/verb divergence, noise floor,
  paired tests, selection filter.
- Explicit distinction: behavioral fidelity is not optimality. exp22 supplies
  the outcome bridge.
- Measurement audit: parser certification, exp19 exact-vs-sampled validation,
  and verifier/oracle gate.

### 4. Behavioral results

#### 4.1 Verbatim extraction survives aggressive compression

- exp8: full 0.65; keep-recent@25% 0.66; summary 0.46 (N=25).
- exp17: 2% still 0.57; 25% 0.68 (N=25).
- Claim: no measurable loss at 25% in this regime; not “25% is better.”

#### 4.2 Rewriting, not information amount, is the boundary

- exp23 matched-budget comparison: raw skeleton+tail 0.71/0.74/0.68;
  keep-recent 0.68/0.66/0.69; pooled difference +0.037, p=0.39.
- Canonical one-liner+tail loses about 13 points; summary loses about 20.
- exp20 supports the pattern: containment rho=-0.23, stronger than NLL.
- Deployable rule: preserve tokens verbatim; choose between verbatim policies
  based on systems cost, not an unsupported quality ranking.

#### 4.3 Off-policy evaluation invents non-transferring laws

- exp4: freeze/block asymmetry strong off-policy, null on-policy (p=0.78).
- exp21: wrapper advantage +18 points off-policy, null/reversed on-policy
  (p=0.36).
- This is the only section allowed to quote off-policy numbers, paired directly
  with their on-policy failures.

### 5. Outcome-grounded live evaluation

exp22, four arms with identical triggers and handoff budgets:

A. keep recent; B. raw skeleton+tail; C. stock summary; D. no compaction.

Primary: valid Pass@1. Secondary: tasks that actually compacted,
post-compaction stalls, compaction count, and D at compaction events. The
verifier must carry a clean marker; infra failures are retried, not scored.
This section is the remaining submission-critical result.

### 6. What did not work

Brief, useful negative results: format effect null on-policy for Qwen N=24;
DPO null at 86-pair dose; rate-curve estimator floor/compute failure. Keep
these scoped and do not turn them into new claims.

### 7. Limitations

One coding-agent domain; one primary local model until outcome row is complete;
25 usable behavioral points; selection conditional on full-context acting;
next-action fidelity can preserve bad behavior; D values cannot be compared
across experiments; task outcome variance; raw-skeleton exactness is defined
relative to each scaffold’s serialization.

## Figure plan

1. **Behavior vs budget:** full, keep-recent, raw skeleton+tail, canonical
   rewrite, summary; exp8/17/23 only.
2. **Same proposal, two regimes:** exp4 and exp21 off-policy effects beside
   on-policy nulls.
3. **Outcome bridge:** Pass@1 by exp22 arm, with compacted-task subset and
   behavioral preservation inset.
4. Appendix: audit timeline and exact-vs-sampled D validation.

## Cut from the main paper

exp3, exp5, exp6 curve, exp7, exp9/10 preliminary runs, exp12, exp13, exp14
GLM cliff, exp15, exp16 training details, exp18, and broad H200 coverage.
exp11 selection can enter the appendix only if exp22 validates the metric
externally.

## Submission decision

A full ICLR claim requires an interpretable exp22 row. If no competent model is
available before the writing deadline, submit as a behavioral-measurement paper
only and state that task utility is unresolved; do not imply a deployable task
success gain.
