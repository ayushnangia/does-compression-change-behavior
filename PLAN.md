# Research plan: behavioral distortion as the reward signal for compaction

## The three problems (and why they are one problem)

**P1 — Credit assignment across repeated compactions.** CompactionRL credits
summaries through the terminal reward with cross-trajectory GAE discounting:
the first summary in a 4-chain trains at 13.5% of the last segment's signal
(exp1, exact, from their own hyperparameters). Their defense: that's correct
temporal discounting. Our attack: **discounting is the wrong prior for
summaries** — an action's influence decays through environment dynamics; a
summary is an information bottleneck, a parent of every downstream state.
Its causal weight does not decay. (exp10 tests this: if one compaction's
damage persists through the real continuation rather than healing, early
summaries have non-decaying influence.)

**P2 — Gains don't transfer to single-window evaluation.** Their own tables:
47.5 -> 43.7 single-window; Long 49.0 vs 59.0. exp5's mechanism: format alone
moves acting by 12 points at constant information — interface familiarity is
learnable and non-transferable. Decomposition to force:
gain = representation quality + loop adaptation, and only the first is
"learning to compress."

**P3 — GRPO / value-function engineering.** exp2: segments-as-samples give
4-compaction rollouts 3.2x gradient mass and inflate advantages +0.37 when
compaction correlates with difficulty. The paper's fix (30B critic, value
pretraining, 2:1 critic updates, length-adaptive lambda, cross-trajectory
GAE) is machinery compensating for a structural mismatch: **trajectory
splitting inside RL**.

**One diagnosis:** all three problems come from crediting compaction through
the terminal task reward. **One fix:** grade each summary immediately, at
the boundary, with behavioral distortion D — dense (no credit path, kills
P1), executor-agnostic and measurable for portability (kills P2), offline
and self-supervised (no critic, no groups, no splitting — kills P3).
Execution keeps vanilla RL on unsplit rollouts with compaction frozen.

## The experiment ladder

| id | experiment | proves | status |
|----|-----------|--------|--------|
| E-A | `exp11_best_of_n.py` — sample N summaries per compaction point, select argmin-D, compare vs random/greedy pick on immediate D, grounded agreement, and teacher-forced downstream behavior | D works as a **selection** signal (thesis in miniature, zero training) | ready |
| E-B | `dpo_train.py` + `evaluate_compressor.py` — DPO a LoRA compressor (Qwen3.5-4B) on D-ranked summary pairs from E-A; evaluate on held-out examples | D works as a **training** signal; the learnable compressor, with no critic/GRPO/rollouts | chained after E-A |
| E-C | `exp12_portability.py` — summaries from model A consumed by executor B (9B x 4B matrix; GLM-4.7-Flash joins when cached) | portable gains = representation; non-portable = co-adaptation (the P2 decomposition, without needing their checkpoints) | ready |
| E-D | trained compressor vs naive on Terminal-Bench 2.0 episodes | end-to-end grounded win | after E-B, uses existing TB2 infra |

Supporting evidence already in hand or in flight: exp1/exp2 (P1/P3
quantified), exp3+floor+entropy (D is a valid measure), exp4 (block
structure), exp5/exp9 (interface effect; D discriminates deployed policies),
exp6 (D is well-behaved across rates), exp7/exp10 (chain decay and
persistence), exp8 (D grounded against logged reality), TB2 (task-level
grounding).

## Milestones

1. **E-A positive** -> "selection by D improves downstream behavior" — the
   minimal viable claim for the SSL objective.
2. **E-B positive on held-out** -> "a compressor trained purely on
   behavioral matching, offline, beats the naive summarizer" — the paper's
   core result without any of its machinery.
3. **E-C portable** -> the trained gains are representation, not
   co-adaptation — the direct counterpoint to CompactionRL's transfer
   failure.
4. **E-D** -> grounded Pass@1 delta.

## Honest limits of the decoupling thesis (read before quoting it)

1. **Ceiling by construction.** Pure D-minimization is self-imitation: it
   can at best recover full-context behavior, never exceed it — and exp4
   shows exceeding it is possible (drop_reasoning beats full on acting).
   CompactionRL's task reward can go beyond the reference; ours cannot.
   Mitigation: asymmetric objective — D combined with grounded agreement
   (exp11 already scores both); treat pure-D as the ablation, not the flag.
2. **We replace one component.** CompactionRL jointly trains execution and
   summarization (execution RL alone: 47.5 -> 50.0 single-window). The DPO
   compressor addresses only summarization, and not compaction *triggering*.
   The honest comparison is: their summary-training component vs ours.
3. **Scale.** ~50 pairs is a proof-of-mechanism (~18 gradient steps). A real
   run needs 1k+ pairs, and the cost is in SCORING, not training: ~200-1000
   A100-hours with the HF scorer, ~10x less after porting scoring to vLLM
   batched inference — that port is the gating engineering task.
4. **Preference noise.** With 8-sample scoring (floor ~0.35), small D-gaps
   are noise. dpo_train filters pairs at min_gap 0.25; expect to discard a
   large fraction.
5. **Off-policy drift.** One-shot DPO trains on the base summarizer's
   distribution; serious versions iterate generate -> score -> train (2-3
   rounds, multiplying cost).
6. **E-D integration.** Terminus-2 has a native compaction hook
   (enable_summarize / proactive_summarization_threshold) and vLLM can serve
   LoRA adapters on the same endpoint — feasible, but real harness surgery.

Revised headline (defensible form): *behavioral distortion gives a dense,
offline, critic-free training signal for the summarization component — at
~0.1% of the RL stack's compute — with a measured validity envelope and a
known parity ceiling.*

## Honest risks

- D at N=8 samples is noisy; best-of-N selection may need more samples per
  summary (mitigate: agreement-with-logged as a second scorer).
- DPO on 4B may co-adapt to the 9B executor used for scoring (mitigate:
  E-C portability check is part of the eval).
- Held-out split is by example index, not repo (upgrade before any writeup).
- Iterated/deep chains and real RL integration are out of scope for this
  ladder; they are the follow-up once E-B exists.
