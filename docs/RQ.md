# The research question (formal, post-hoc sharpened — before any external writing)

## The question

> **When an agent's context is compressed, what determines whether its
> BEHAVIOR survives — and can behavioral survival itself serve as the
> compression objective?**

## The knowledge claims (each falsifiable, each with its evidence scope)

1. **Information preservation and behavior preservation are different
   objectives, and current compressors optimize the wrong one.**
   Evidence: summaries (information-optimized) sit on the worst behavioral
   D(R) curve; truncation beats them at every tested rate.
   *Scope: SWE-agent traces, 4k contexts, Qwen3.5 family; on-policy
   replication in flight.*

2. **Behavioral value is concentrated in the agent's action history, not in
   the informational bulk.** Deleting observations (52% of tokens) is
   behaviorally free; deleting tool calls (29%) is catastrophic.
   *Scope: two model families, 5 runs, matched budgets.*
   Surprising against the token-importance prior: high-information blocks
   have low behavioral value.

3. **Behavioral divergence is measurable (above a sampling floor),
   externally valid (tracks agreement with real logged actions), and
   usable as a dense offline signal** (best-of-N selection p=0.0004;
   DPO training run in progress).
   Consequence if it holds: the summary-training component of RL-with-critic
   pipelines (CompactionRL) is replaceable at ~0.1% of the compute, and the
   compaction credit-assignment problem dissolves (graded at the boundary).

4. **The delivery format of compressed context is a first-order variable**,
   ranging from measurable degradation (Qwen, 12 pts) to total collapse
   (GLM). *Scope: being hardened by exp14's 3-model matrix.*

## Against the four criteria

- **Surprising:** claims 1 and 2 contradict deployed practice and the
  token-importance prior; RELATED.md audit found no direct precedent
  (with the stated search-depth caveat).
- **Fruitful:** immediate engineering consequence (every agent framework's
  compaction changes); opens a program (behavioral-SSL compressors,
  format-native compaction, block-aware budgets, marginalized targets).
- **Foreclosing:** floors, paired stats on pre-registered endpoints,
  padding/wrapper/random controls, cross-family + cross-scale + on-policy
  replication, external grounding, AUDIT.md ledger. Remaining known holes:
  single trace domain, N<=48, TV estimator coarseness.
- **Feasible:** demonstrated — the entire program ran on 1-3 A100s with
  an offline cluster; ambition was cut twice, correctly (RL reproduction ->
  offline DPO; Pass@1 grounding -> logged-action grounding).

## Process debt acknowledged

This document was written AFTER execution (the essay's phase 2 was
inverted). The claims above emerged from exploration rather than preceding
it. Recorded so the paper phase starts from an explicit question — and so
the next project runs phase 2 first.
