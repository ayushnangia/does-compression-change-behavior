# Does compressing an agent's memory change what it does?
### Research showcase — behavioral rate–distortion for agent context compression

**One sentence:** we treat agent-context compression as a rate–distortion
problem where distortion = *behavioral change*, built and validated the
measurement instrument, and used it to find design laws that beat
deployed compaction policies — including two public self-corrections.

**Claim scope:** all findings are on agentic SWE traces (≤4k contexts,
Qwen3.5/GLM families) unless stated. Exact per-claim scopes: `RQ.md`.
Novelty audit against prior art: `RELATED.md`.

All figures in `experiments/figures/`. Claims audit with per-claim confounds
in `AUDIT.md`. Stats: floor-referenced, bootstrap CIs, paired permutation
tests on pre-registered primary endpoints (N=14–43 per cell; exploratory
grade, honestly labeled).

---

## 1. The problem
Every long-horizon agent (Claude Code, OpenHands, Terminus) compresses old
context, usually with a model-written summary. Quality is checked with
perplexity/QA probes. **Nobody checks whether the agent still does the same
thing.** We measure exactly that: acting rate, next-action change vs a
sampling floor, action entropy, and agreement with the action the original
agent actually took (external ground truth mined from the traces).

## 2. Validity of the measure (figs 2, 8)
- **Externally grounded** (fig 8): behavior change tracks disagreement with
  the *real* logged action almost perfectly — full context: 0.73 agreement;
  truncation: 0.67; summaries: 0.30. High D = the agent stops doing what it
  actually did.
- **Target stability** (fig 2): stable at moderate lengths; the padding
  control showed the drift at short lengths is mostly *positional*, not
  informational (padded 0.56 vs truncated 0.60) — **a thread critique we
  tested and confirmed against ourselves**. Consequence: marginalize the
  SSL target over lengths.
- Cross-scale (9B vs 4B): smaller model is noisier (floor 0.44 vs 0.36) but
  not disproportionately less stable.

## 3. The block law (figs 3, 9) — main design finding
Role-aware ablation at equal 50% budget, replicated on Qwen3.5-9B AND
GLM-4.7-Flash, effect stable across 5 runs / 2 seeds / N up to 43:
- deleting **tool-call history** is catastrophic (acting 0.81→0.53, halting
  2.5x) — it is ~29% of tokens;
- deleting **observations** — 52% of all tokens — costs nothing beyond
  random deletion (change 0.50 vs 0.51);
- reasoning text (6%) is mildly load-bearing (0.61).

**Turning the law into a prompt beats every deployed-style compaction
policy** (fig 9): "copy tool calls and paths verbatim, compress the rest"
gets acting 0.80 vs 0.51 (naive) and real-action agreement 0.55 vs 0.33–0.40
for Claude-Code-style / OpenHands-style prompts. Measurement → design, no
training.

## 4. The upset (figs 5, 7)
On agentic SWE traces at 4k contexts with Qwen3.5-family models, across
every budget tested (12.5–75%), **model-written summaries sit on the worst
rate–distortion curve — plain "keep recent tokens" beats them everywhere**
(e.g., 0.49 vs 0.68 change at 25% budget; keep_recent acting 0.94 ≈ full).
Within this scope, the deployed default is the worst option we measured.
Convergent with (but at a different level than) KV-recency results
[StreamingLLM]; on-policy replication in flight; other domains untested —
see RELATED.md for the novelty audit and RQ.md for exact claim scopes.

## 5. D as an optimization signal (fig 11)
Best-of-8 summary selection by D (winner's-curse-guarded, fresh-scored):
immediate D 0.61 vs 0.79, **real-action agreement 0.55 vs 0.38**, downstream
D directionally better. The metric is a usable dense reward — the basis for
the offline DPO compressor (training now; pairs gap-filtered above noise).
This replaces RL-with-critic machinery for the summary component at ~0.1%
of the compute — with a known parity ceiling (see PLAN.md limits).

## 6. Self-corrections (figs 6, 10, 13) — read these first, they buy the rest
1. **Compaction damage HEALS** (fig 10): change 0.86 → 0.47 and acting
   0.43 → 0.90 within ~2 real steps of teacher-forced continuation. Our
   earlier "front-loaded damage needs front-loaded credit" argument is
   weakened: single-step D *overstates* long-run harm (teacher-forcing =
   upper bound on healing; free-running variant scheduled).
2. **The v1 block experiment was mislabeled** (no `<observation>` tags in
   the data; "drop_observations" was a no-op). Caught via an anomaly
   (condition == control exactly), fixed role-aware, rerun, replicated.
3. **The deletion-manifest idea** ("tell the agent what was deleted and how
   to retrieve it") — mostly null at this scale (fig 13): +4 pts acting,
   +8 pts lookup-share, no gain combined with a summary. Possibly masked by
   the recovery harness advertising retrieval anyway; clean retest designed.

## 7. Grounded task-level evaluation (Terminal-Bench 2.0)
Full offline Harbor+Terminus pipeline on Narval (89 tasks, Apptainer images
pre-baked, vLLM-served Qwen3.5 models): pipeline validated end-to-end; at
benchmark-default 15-min budgets both models score 0/50+ (41/78 trials
timeout; tasks carry 3-hour expert estimates) — a valid datum with no
discriminative power. Rerun in flight: 25 easiest tasks at 4x timeout to
land mid-range where compression deltas are measurable.

## 8. CompactionRL analysis (fig 1)
From the paper's own equations: the first summary in a 4-chain trains at
**13.5%** of the last segment's signal; GRPO with segments-as-samples gives
heavily-compacted rollouts **3.2x** gradient mass. Combined with fig 10
(healing), the credit story is genuinely open — our dense-boundary-reward
alternative dissolves rather than answers it, and its first test (DPO) is
running.

## 9. Related work
CompactionRL (arXiv:2607.05378); rate–distortion (Shannon 1959) and
information bottleneck (Tishby et al. 1999); long-context degradation (Liu
et al. 2024; RULER, Hsieh et al. 2024); agent memory (Reflexion, MemGPT,
generative agents, context folding); GAE/PPO/GRPO (Schulman et al.; Shao et
al. 2024); SWE-bench Verified; Terminal-Bench 2.0 + Harbor;
nvidia/Open-SWE-Traces.

## 10. Limitations (complete list in AUDIT.md)
Single trace source written by a different model (on-policy replication
planned — our own TB2 trajectories are already collected); N=14–43, 1–2
seeds; TV-from-8-samples estimator (floor-referenced); conditional on
decision points where the model acts; exploratory-grade inference with
pre-registered primary endpoints only.

## Reproduce
```bash
python tests/run_tests.py                      # 59 checks
python run_experiment.py --examples-file examples_prefetched.json
sbatch experiments/corrected_rerun_job.sh      # the headline experiments
python experiments/make_figures.py             # all 12 figures
```
