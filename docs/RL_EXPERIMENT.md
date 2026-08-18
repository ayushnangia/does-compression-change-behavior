# exp24 — proper behavioral-reward training experiment

Status: designed, **not run**. Starts only after exp22 Gate 1 demonstrates a
non-floor task-success evaluator.

## What has and has not been done

- exp1/2 are analyses of credit weighting and GRPO assumptions, not RL runs.
- DPO round 1 was invalid and is retracted.
- T4 is the one valid training run: 86 on-policy pairs, LoRA, held-out
  evaluation; DPO 0.73 vs base 0.75, p=0.345. This is a scoped null, not a
  verdict on learning.
- No PPO, GRPO, or TRPO optimizer has been run in this project.

## Question

Can direct optimization of a compactor for downstream behavioral preservation
outperform supervised and preference-learning baselines, and does any gain
transfer to live task success?

## Why GRPO, not an algorithm zoo

GRPO naturally matches the data-generating unit: generate a group of candidate
compactions for one history, execute each under the same downstream model, and
normalize rewards within that history. PPO adds a critic whose quality becomes
another confound; TRPO adds substantial implementation cost without addressing
the reward-validity question. One credible GRPO comparison is stronger than
five underpowered optimizers.

## Frozen design

### Data

- Build at least 1,000 compaction points from our own trajectories.
- Split by Terminal-Bench task/repository **before** candidate generation:
  70% train, 15% validation, 15% test.
- No decision points from one task may cross splits.
- Prefer trajectories from a model with nonzero valid Pass@1.

### Models

- Compactor: one trainable 4B–9B model, LoRA first; model generates its own
  candidates (no off-policy 9B-text/4B-trainee mismatch).
- Executor: fixed model/scaffold used by exp22.
- Freeze tokenizer, chat template, sampling parameters, context budget, and
  model revisions.

### Four matched arms

1. **Base:** untrained compactor.
2. **SFT-best:** supervised fine-tuning on the lowest-distortion candidate.
3. **DPO:** preference pairs from best-vs-worst candidates, gap-filtered.
4. **GRPO-D:** group size 4, direct downstream behavioral reward.

All arms receive the same train histories, candidate budget, LoRA rank,
optimizer-token budget, and evaluation draws.

### Reward

For candidate compression `c` at history `h`:

`R = logged_action_agreement(c) - alpha * halt_increase(c) - beta * budget_violation`

Use multiple executor samples and a cached full-context reference. Do not use
raw sampled D alone: exp19 shows tool-level D is floor-dominated, and optimizing
the evaluation metric directly invites circularity. Agreement and halt are
computed downstream, not from summary text.

### Primary endpoint

Held-out **logged-action agreement** on unseen tasks, with paired bootstrap CI
and a predeclared GRPO-D vs DPO comparison. Secondary: halt rate, exact/tool
agreement, and output-token cost.

### External endpoint

Plug the best frozen compactor and base into exp22 on the same task subset.
Report valid paired Pass@1. This is required before saying training improves an
agent rather than merely its behavioral proxy.

## Power and stop rules

- Pilot: 100 train points, 20 held-out, used only to verify reward/gradient
  plumbing; no claims.
- Main: >=1,000 train points and >=150 held-out points across unseen tasks.
- Two training seeds minimum.
- Stop if reward accuracy on known best/worst controls is <70%, if held-out
  acting is at floor, or if no arm changes the validation endpoint after the
  predeclared token budget.
- Never add PPO/TRPO after a null without first diagnosing reward validity.

## Required artifacts

Split manifest, candidate/reward JSONL, exact training configs, optimizer
curves, adapter hashes, per-example held-out arrays, paired tests, and exp22
outcome rows. All must be committed or copied from purgeable scratch.

## Place in the paper

This is an ICLR-strength extension if it completes cleanly. It is not required
to state the current measurement finding. If time is insufficient, report the
86-pair DPO null in the appendix and do not market the paper as RL.
