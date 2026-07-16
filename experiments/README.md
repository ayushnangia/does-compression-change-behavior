# Claim-verification experiments (exploratory)

Status: **experimental / for group discussion — not paper-grade.** Each
experiment targets one claim from the compression-behavior thread. exp1-exp2
are analytic (no GPU, run anywhere); exp3-exp5 are behavioral (1x A100,
inference-only, single model, small N — read them as existence proofs and
effect-size estimates, not confirmations).

| # | Claim | Experiment | Status |
|---|-------|------------|--------|
| 1 | Earlier summaries get a much weaker learning signal as the compaction chain deepens | `exp1_credit_decay.py` — exact attenuation under the paper's cross-trajectory GAE (lambda = 1-1/(1.5 l), gamma = 1) | **done** — see results below |
| 2 | GRPO-style groups are distorted by variable segment counts | `exp2_grpo_weighting.py` — Monte-Carlo of segment-expanded group normalization | **done** — see results below |
| 3 | Full-context behavior target may be length-unstable (thread critique) | `exp3_target_stability.py` — length sweep vs sampling floor | **done** — see results below |
| 4 | Flat next-action KL averages over heterogeneous block losses | `exp4_block_ablation.py` — block-selective deletion at matched budget + failure-kind profiles | **done** — see results below |
| 5 | CompactionRL gains partly = "operating inside the compaction loop" | `exp5_format_vs_content.py` — same summary content, varied interface format | **done** (proxy only) — see results below |

## Results so far

### exp1 — credit attenuation (analytic, paper's own hyperparameters)

Signal scale of a segment's summary tokens relative to the final segment:

| chain length K | seg 1 | seg 2 | seg 3 | seg 4 |
|---|---|---|---|---|
| 2 | 51.3% | 100% | | |
| 3 | 26.4% | 51.3% | 100% | |
| 4 | **13.5%** | 26.4% | 51.3% | 100% |

Two takeaways: (a) the claim is quantitatively real — the first summary in a
4-chain trains at ~1/7 the strength of the last segment; (b) it is **by
design**, not an oversight: the paper's correction deliberately restores
temporal distance. The attenuation per segment-step (x0.513) is
**scale-invariant** (independent of segment length; depends only on alpha).
So the sharp version of the argument is: *discounted credit is the wrong
prior for summaries specifically*, because a summary gates all downstream
information — its causal weight does not decay like an ordinary action's.
The paper rejects auxiliary summary rewards, so this attenuation is
uncompensated.

### exp2 — GRPO segment-expansion bias (Monte-Carlo, G=8)

- Gradient mass scales with segment count: K=4 rollouts carry **3.2x** the
  gradient mass of K=1 rollouts for the same outcome (K independent case).
- In the realistic regime (more compactions on harder, more-failing tasks),
  the group mean is dragged toward heavily-compacted outcomes: successful
  rollouts' advantages inflate by **+0.37** on average.

Supports both the claim and the paper's move to PPO, and gives the concrete
mechanism for "credit assignment becomes hard to interpret under GRPO."

### exp3 — target stability (Qwen3.5-9B, 14 usable examples, floor 0.357)

| context length | drift vs 4096 | x floor |
|---|---|---|
| 2048 | 0.375 | 1.1 (within noise) |
| 1024 | 0.562 | 1.6 (real drift) |

The full-context target is **length-stable at moderate truncation and
unstable under aggressive truncation**. Verdict on the thread critique:
the SSL target is usable as-is when the retained context is long enough;
near the truncation edge, marginalize the target over lengths/layouts.

### exp4 — block ablation (equal 50% budget, floor 0.357)

| variant | acting | change | none | lookup | commit |
|---|---|---|---|---|---|
| random_control | 0.89 | 0.62 | 0.11 | 0.43 | 0.46 |
| drop_tool_calls | **0.67** | 0.69 | **0.33** | 0.38 | 0.29 |
| drop_observations | 0.87 | 0.62 | 0.13 | 0.43 | 0.44 |
| drop_reasoning | **0.93** | **0.45** | 0.07 | 0.42 | 0.51 |

At identical token budgets: deleting **tool-call history triples halting**;
deleting **reasoning text is less damaging than random deletion** (closest
to full-context behavior, acting rate even improves). Different blocks
carry very different behavioral value per token — a flat divergence
averages this away. Claim supported. Design corollary for compressors:
preserve action history verbatim; reasoning prose is nearly free to drop.

### exp5 — format vs content (identical summary in every row, floor 0.357)

| format | acting | change | none | lookup | commit |
|---|---|---|---|---|---|
| inline_note | 0.53 | 0.82 | 0.47 | 0.26 | 0.27 |
| resume_template | **0.50** | **0.88** | 0.50 | 0.27 | 0.23 |
| assistant_voice | 0.56 | 0.82 | 0.44 | 0.37 | 0.20 |
| bare | **0.62** | **0.74** | 0.38 | 0.33 | 0.29 |

With information content held constant, format alone moves acting by 12
points and behavior change by 14 points. The paper-style resume template
is the WORST format for an untrained model — consistent with the claim
that part of CompactionRL's gain is learning to operate its own compaction
interface (and with why those gains do not transfer to single-window
evaluation). Also note all formats sit far above the floor: summaries
change behavior a lot regardless of wrapper.

## Round 2: issues raised in the thread → experiments & resolutions

| Issue (who) | Experiment / resolution | Status |
|---|---|---|
| "Preserve the uncertainty structure, not just the argmax" (Paras, Anton) | `metrics.action_entropy` — entropy of the sampled action distribution vs full-context entropy; reported in exp7 | metric added |
| "Verify the full-context oracle isn't already lossy; model choice matters" (Anton) | exp3 rerun at a second scale (Qwen3.5-4B vs 9B): if the target destabilizes as capacity shrinks, oracle quality is a real budget item | queued (exp7_job) |
| Chain-depth credit assignment presumes damage propagates (us) | `exp7_compaction_chain.py` — iterated summarization: D(k), acting, entropy for k=1..3 compactions. Compounding D(k) = behavioral basis for uniform (undiscounted) summary credit; flat D(k) = the concern is overstated. Also probes why CompactionRL caps at 3 compactions | queued |
| "4B-on-AIME confound: capacity vs context" (Anton) | design (not built): padding control — same problem + irrelevant padding to force compaction. Fails padded but solves unpadded ⇒ context problem (compaction can help); fails both ⇒ capacity problem (compaction can't). Attribute compressor gains only on the first set | designed |
| exp1's attenuation is by design; the fix is contested | resolution options (need RL infra to test): uniform λ for summary tokens, summary-specific credit, or auxiliary summary reward (paper rejects the last). exp7's D(k) decides whether any fix is warranted | analysis |

Note on iterated summarization: real chains interleave NEW interaction between
compactions; exp7 isolates the pure re-compression decay (a lower bound on
chain damage). Stated in the script docstring too.

## How to run the GPU set

```bash
# once, login node:
module load gcc arrow && source ~/ENV-compress/bin/activate
python prefetch.py --num-examples 24 --context-tokens 4096 --recent-tokens 512

# then:
sbatch experiments/narval_exp_job.sh
```

Results land in `experiments/results/*.json` and stdout tables in `exp_<jobid>.out`.

## How to read exp3-exp5

- **exp3**: `drift vs max >> floor` -> the SSL target imports length
  artifacts; marginalize the target over lengths/layouts before trusting it.
  `drift ~ floor` -> the thread critique is empirically void in this regime.
- **exp4**: equal budgets, unequal damage and different none/lookup/commit
  profiles -> flat divergence conflates distinct information losses ->
  block-aware distortion measure is justified.
- **exp5**: behavior moves with format at constant content -> part of
  compaction performance is interface familiarity, which is learnable by
  RL-in-the-loop and would NOT transfer to single-window eval — consistent
  with the paper's own single-window regressions (47.5 -> 43.7 on
  GLM-4.7-Flash; ablation Long column 49.0 vs 59.0).

## Known limitations

- Single model (Qwen3.5-9B default), off-policy traces
  (nvidia/Open-SWE-Traces), single-step behavior, N=16 examples.
- exp5 is an inference-level proxy for a training-level claim.
- Action parsing is regex/heuristic (`behavior.py`); treat small deltas
  (< 2x floor) as noise.
