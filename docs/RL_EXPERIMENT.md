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

Can direct optimization of an **extractive selector** for downstream
behavioral preservation outperform fixed extractive heuristics, supervised
and preference-learning baselines, and does any gain transfer to live task
success? The selector emits block indices; the system copies those trace
blocks byte-for-byte. The policy cannot paraphrase or hallucinate memory.

## Why GRPO, not an algorithm zoo

GRPO naturally matches the data-generating unit: at one fixed history, generate
a group of block selections, execute each under the same frozen downstream
model, and normalize rewards only within that history. This is a contextual
bandit, so **GAE is removed rather than modified**. PPO adds a critic whose
quality becomes another confound; TRPO adds implementation cost without fixing
reward validity. Dr-GRPO removes completion-length bias, sequence-level
importance weights match the sequence-level action, and each history enters
once—no segment-expanded gradient mass.

## Frozen design

### Data

- Build at least 1,000 compaction points from our own trajectories.
- Split by Terminal-Bench task/repository **before** candidate generation:
  70% train, 15% validation, 15% test.
- No decision points from one task may cross splits.
- Prefer trajectories from a model with nonzero valid Pass@1.

### Models

- Compactor: one trainable 4B selector, LoRA first; it generates its own
  index selections (no off-policy 9B-text/4B-trainee mismatch).
- Executor: frozen `Qwen/Qwen3.8-27B` bf16 under the exact exp22 scaffold.
  Qwen3.5-9B may be used only for a non-result plumbing smoke.
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

For candidate selection `c` at history `h`:

`R = .70 tool_agreement + .20 verb_agreement + .10 acting_rate`

Invalid JSON and budget overflow receive a hard negative reward. Use multiple
executor samples. Do not use raw sampled D alone: exp19 shows tool-level D is
floor-dominated. All reward terms are computed from the frozen executor’s
next actions, never from memory text. The reward is frozen before training.

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

## Implementation

- `exp24_data.py`: exact turn extraction, strict index parser, verbatim renderer.
- `exp24_prepare.py`: deterministic task-level split and selector manifests.
- `exp24_credit.py`: group advantages, unit trajectory mass, frozen reward.
- `exp24_grpo_train.py`: LoRA Dr-GRPO with a frozen vLLM executor reward.
- `exp24_job.sh`: supported two-H100 Slurm launcher (GPU0 train, GPU1 execute).

The old 64-example Qwen3.5-derived data yielded only 31/8/13 rows and is
**forbidden for training this lineage**. After the Qwen3.8 baseline, an
efficient turn-boundary-aware builder harvests up to 2,000 decision points
from Qwen3.8's own trajectories and splits by task. The trainer hard-fails if
any row's `source_model` is not `Qwen/Qwen3.8-27B`. The main run still requires
>=1,000 points; smaller data is plumbing only.

## Required artifacts

Split manifest, candidate/reward JSONL, exact training configs, optimizer
curves, adapter hashes, per-example held-out arrays, paired tests, and exp22
outcome rows. All must be committed or copied from purgeable scratch.

## Place in the paper

This is an ICLR-strength extension if it completes cleanly. It is not required
to state the current measurement finding. If time is insufficient, report the
86-pair DPO null in the appendix and do not market the paper as RL.
