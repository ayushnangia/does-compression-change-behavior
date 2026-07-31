# Literature acquisition plan (what backs each section, what is owed)

RELATED.md (this dir) has the per-claim novelty audit at keyword depth.
This file is the ACQUISITION list: what must be read/cited per paper
section, and the citation-graph walk targets. Caveat inherited from
RELATED.md: everything so far is arXiv keyword search; the walk is owed
before any external novelty claim.

## §3 The instrument — theory lineage (NEW, from the TD/RL discussion)

The load-bearing chain: compressed context = approximate information
state; per-step policy TV bounds outcome loss; halting = absorbing
deviation = the bound's worst case.

| cite | role in the paper |
|---|---|
| Kakade & Langford 2002 (CPI) | performance-difference lemma: per-step TV bounds outcome gap — the a-priori license for D |
| Schulman et al. 2015 (TRPO, Thm 1) | the TV-form bound we instantiate; already name-dropped, upgrade to theorem |
| Ross, Gordon & Bagnell 2011 (DAgger) | compounding error O(T^2 eps): grounds D@k / exp10 persist-vs-heal |
| Li, Walsh & Littman 2006 | state-abstraction taxonomy: compaction = online state abstraction |
| Ferns, Panangaden & Precup 2004/2011 | bisimulation METRICS — TV between action-conditioned outcomes; D is the policy term. HIGHEST prior-art risk: walk this citation graph first |
| Castro 2020; Zhang et al. 2021 (DBC) | deep-RL bisimulation revival — reviewers will know these |
| Subramanian & Mahajan (2022, JMLR) | approximate information states: value-loss bounds for approximate sufficient statistics of history — the cleanest POMDP framing of compaction |
| Sutton 1988; Sutton & Barto ch.12 | TD(λ)/eligibility traces — one paragraph framing CompactionRL's credit decay (exp1) as a known-bad λ choice |
| Farahmand 2011 / error-propagation in ADP (optional) | if a reviewer pushes on multi-step propagation formalism |

## §2 Related work — compression & agents (have, from RELATED.md)

LLMLingua-1/2 (information-centric compression — the objective we invert);
CompactionRL 2607.05378 (the deployed motivation); StreamingLLM 2309.17453 +
H2O (KV recency, convergent); agent-memory survey 2606.24775 (independently
documents the evaluation gap); DAST 2502.11493 (token importance); Parallel
Context Compaction 2605.23296 (summaries lossy, serving-side); lost-in-the-
middle; RULER; Reflexion/MemGPT/context-folding (heuristic memory);
SWE-bench; Terminal-Bench 2.0/Harbor; Shannon 1959 + Tishby IB (rate-
distortion framing: we change the distortion measure to behavioral).

## §3 metric adjacents (have, partially)

Semantic entropy / Kernel Language Entropy (semantic clustering upgrade —
cite as planned extension for limitation #2); MMD two-sample tests; TRPO
policy distance (above); imitation-learning action matching (grounded
agreement); Zhang & Khattab 2026 harness-generalization blog (input-side
metrics; our exp20 is the quantitative bridge — check if citable form
exists by deadline).

## Citation-graph walk — ordered targets (the owed work, ~1 day)

1. **Ferns bisimulation metrics** — forward citations for "behavioral
   metric" + "context/history compression". The one place a direct
   precedent for D could hide.
2. **Subramanian AIS** — forward citations for LLM-agent applications
   (2025–26). If someone already framed compaction as AIS, we cite and
   differentiate on measurement + laws, not framing.
3. **LLMLingua-2 forward citations, 2025–26 agent slice** — any
   "compression for agents" paper evaluating beyond end-to-end scores.
4. **CompactionRL's own related-work section** — whatever they position
   against, we must know.
5. **Agent-memory survey (2606.24775) reference list** — mine it; it is a
   pre-built map of the memory-evaluation gap we fill.
6. Search terms for the final sweep: "behavior preservation compression",
   "action distribution divergence agent", "context compaction evaluation",
   "summarization agent trajectory divergence" on arXiv 2025–2026.

## Semantic Scholar note

Rate-limited from cluster (RELATED.md) — run the walk from a non-cluster
machine or with an API key.
