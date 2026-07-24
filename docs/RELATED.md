# Related work & novelty audit (claim -> closest prior art -> distinction)

Method: arXiv API search per claim + abstract-level reads of the nearest
neighbors (2026-07-15). Caveat: keyword search, not citation-graph walking —
a full review pass is required before paper submission. Semantic Scholar was
rate-limited from the cluster; re-run from elsewhere for citation counts.

## Claim 1 — Behavioral distortion as the compression objective/metric

**Closest prior:** LLMLingua-2 (arXiv:2403.12968) — task-agnostic prompt
compression; explicitly criticizes information-entropy as "not aligned with
the prompt compression objective" but replaces it with *token-importance
distillation*, still information-centric, evaluated on QA/summarization
tasks, not agents.
**Independent gap confirmation:** "Are We Ready For An Agent-Native Memory
System?" (arXiv:2606.24775) states existing memory evaluations use only
end-to-end task metrics, "treating the underlying system as a monolithic
black box" — i.e., a 2026 survey confirms nobody measures what memory
operations do to *behavior*.
**Our distinction:** distortion = next-action distribution divergence of the
consuming agent, floor-referenced and externally grounded against logged
actions. Not found in any searched work.

## Claim 2 — Truncation beats model-written summaries (behavioral D(R))

**Closest prior:** StreamingLLM / attention sinks (arXiv:2309.17453) —
recency (+ sink tokens) dominates for KV-cache eviction. Convergent evidence
that recent context carries most of what matters, but at the *attention/
serving* level, not text-level compaction, and with no summary comparison.
**Convergent motivation:** Parallel Context Compaction (arXiv:2605.23296)
observes LLM summarization is "inherently lossy" with retention that
"fluctuates substantially from run to run" — but their contribution is
serving latency, not measurement or comparison.
**Our distinction:** a head-to-head rate-distortion comparison of
summary vs truncation vs random vs block-aware ON AGENT BEHAVIOR appears
novel. Scope honestly: shown on SWE-agent traces at 4k contexts.

## Claim 3 — Block-level behavioral value (tool calls >> observations)

**Closest prior:** selective/token-importance compression (LLMLingua line,
DAST arXiv:2502.11493) rank tokens by importance — but importance is
information-theoretic, computed for text generically, and none decompose
*agent-trace structure* (actions vs observations vs reasoning) or measure
behavioral consequences per block type.
**Our distinction:** role-aware ablation with matched budgets + behavioral
readout; the "actions are the backbone, observations are bulk-droppable"
law appears novel. (And is mildly SURPRISING against the token-importance
prior: observations have high information content but low behavioral value.)

## Claim 4 — Dense offline behavioral reward replaces RL-for-compaction

**Closest prior:** CompactionRL (arXiv:2607.05378) — the RL-with-critic
approach we analyze; context folding / memory papers (Reflexion, MemGPT,
generative agents) are inference-time heuristics without trained objectives.
**Our distinction:** best-of-N selection + offline DPO on D-ranked pairs;
no rollouts, no critic, no group normalization. The E-A result (selection
works, grounded, p=0.0004) has no found precedent.

## Claim 5 — Interface fragility (format alone moves/destroys behavior)

**Closest prior:** prompt-sensitivity literature (format robustness studies
exist for classification tasks); nothing found on compaction-note format
specifically, and nothing on full degeneration (GLM collapse).
**Our distinction:** same-content different-wrapper design in the compaction
setting, cross-family, with a collapse case. exp14 (running) hardens it.

## Known prior art we must cite as context (not competition)
Shannon 1959 (rate-distortion); Tishby et al. 1999 (IB); Liu et al. 2024
(lost in the middle); RULER; StreamingLLM/H2O (KV recency); LLMLingua 1/2;
CompactionRL; Reflexion/MemGPT/generative agents/context folding; SWE-bench;
Terminal-Bench 2.0/Harbor; Open-SWE-Traces.


## Metric lineage (added after the estimator survey)

Our measurement constructs are instances of established families, not inventions:
- **TV between action distributions** = the policy-distance of trust-region RL
  (TRPO's monotonic-improvement bounds are stated in total variation;
  Schulman et al. 2015). We apply it to LLM-agent next-action distributions.
- **Agreement with logged actions** = action-matching accuracy, the standard
  imitation-learning evaluation.
- **Exact logprob-scored next-tool distribution** (ours, implemented): removes
  sampling noise entirely at tool granularity; closed candidate-set assumption.
- **Adopted-as-planned upgrades from the UQ/statistics literature:**
  semantic clustering of sampled outputs before distribution comparison
  (semantic entropy family: Farquhar et al.; Kernel Language Entropy,
  arXiv:2405.20003) for the action-equivalence problem; MMD two-sample tests
  over action embeddings (unbiased small-sample estimator) as robustness check.
- Rejected with reason: token-level KL (measures phrasing, not behavior).

Caveat unchanged: keyword-level search; citation-graph walk still required
before external claims of metric novelty.


## Convergent work: harnesses as compositional generalizers (Zhang & Khattab 2026)

Blog (alexzhang13.github.io/blog/2026/harness/): recursive/offloading harnesses
generalize because context offloading keeps each LM call "locally
in-distribution"; they measure eval-vs-training trajectory closeness with
input-side surface metrics (Levenshtein, n-gram containment/Jaccard) and
explicitly note the missing piece: a principled distance on OUTPUT
distributions with semantic awareness. Relationship to us:
- our behavioral D is exactly that output-side measure (their delta-ball /
  epsilon-ball sketch = their input metrics + our output metric);
- their in-distribution mechanism explains WHY compaction should work; our
  format results (exp5/14/15) specify when it fails (format-OOD interfaces);
- our block law + format collapse jointly refute their smoothness assumption
  ANISOTROPICALLY: content directions are nearly flat (36% of tokens
  deletable), format directions are cliffs (constant content, total collapse);
- testable bridge (planned): correlate D with an in-distribution proxy of the
  compacted context (perplexity or their n-gram containment vs natural trace
  contexts) - would unify the two frameworks quantitatively.

## Novelty risk register
- Keyword search only — MUST citation-walk LLMLingua-2 and 2606.24775
  forward-references before external claims.
- "Truncation is a strong baseline" may exist informally in engineering
  blogs/ablations of memory papers — soften to "systematically shown" not
  "first shown".
