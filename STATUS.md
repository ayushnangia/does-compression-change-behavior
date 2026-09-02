# STATUS — current truth only

Last updated: 2026-08-18. Detailed experiment disposition:
`docs/EXPERIMENT_LEDGER.md`. Execution plan: `docs/ICLR_PLAN.md`. Historical
verdicts and retractions: `docs/AUDIT.md`.

## Current state

- Active durable chain: r8 gap collection 840989 → rebuild 840990 → integrity
  gate 841033 → powered Dr-GRPO seeds 42/43 (841034/841035). Full-node jobs
  release only if the rebuilt train split reaches ≥1,000 and passes lineage,
  SHA, chat-shape, and task-disjointness checks.
- Repository/data: on-policy dataset (64 examples, 2k–209k tokens) and all
  Aug 4–6 powered result artifacts are committed.
- Citable HF package staged: 132 byte-exact valid Qwen3.8 trajectories with
  checksums, index, provenance/exclusions, limitations, and BibTeX. Remote
  push is blocked only by a missing write-scoped `HF_TOKEN`; put it in
  gitignored `.env`, never chat.
- Test gate: **144/144** after exp22/exp24 lineage, budget, parser and
  generation-default guards.
- Paper policy: on-policy results only; off-policy appears only as evidence
  that evaluation regimes do not reliably transfer.

## What is established

1. **Verbatim recency is nearly free at 25%.** exp8 powered N=25: logged-action
   agreement full=0.65, keep-recent@25%=0.66, summary=0.46.
2. **A tiny extractive core remains useful.** exp17 powered N=25: agreement
   0.57 at 2%, 0.68 at 25%.
3. **Do not rewrite action history.** exp23 powered N=25: raw skeleton+tail and
   keep-recent statistically tie (pooled p=0.39); canonical rewriting loses
   about 13 points and summaries about 20.
4. **Off-policy laws did not transfer.** exp4 freeze law is null on-policy
   (0.35 vs 0.28 halts, p=0.78); exp21 wrapper effect is null/reversed (0.49
   vs 0.55, p=0.36).
5. **Containment is supporting, not a law.** exp20 powered rho=-0.23 on-policy,
   stronger than NLL (-0.12), but predictive only.
6. **Qwen format phenotype is null on-policy.** exp14 completed Aug 8, N=24:
   native=0.47 vs wrapper=0.48 at top_p=1. The GLM cliff remains off-policy
   and is not a paper headline.
7. **The 25/64 acting filter is not a long-context collapse.** Usable rates
   are 0.20/0.40/0.50/0.50 across <16k/16–32k/32–64k/64k+ bins; usable
   histories are longer on average, but not significantly (p=0.254).

## What failed or is retired

- exp6 powered rate curve timed out twice at 24h. Checkpointing saved N=16 but
  had no resume mechanism, so the second run repeated work. The partial floor
  is 0.45 [0.34,0.56] and policy ordering is unstable. **Retired from the
  critical path**; exp8/17/23 answer the central budget question more cleanly.
- exp3 is floor-dominated (0.68); exp12 has N=1; exp13 is confounded; exp15 is
  format-confounded; exp16/DPO is a clean null at this dose. None blocks the
  paper.
- Every TB2 score before Aug 6 is retracted: verifiers could not run offline.

## Validity gate

- Offline verifier smoke: **GREEN** (real pytest execution and reward file).
- Oracle/reference solution: **GREEN on 2026-08-18**. The reference solution
  for `break-filter-js-from-html` passed offline and wrote reward=1.
- Reproducible gate: `bash tb2/oracle_smoke.sh`.
- easy25 images: 25/25 rebaked with system-path uv caches.
- Remaining 64 images are stale; this does not block easy25 Gate 1, but they
  must be rebaked before an all-89 run.

## Sole next objective

**exp22: connect behavioral preservation to valid Terminal-Bench task
success.** Final lineage is `Qwen/Qwen3.8-27B` bf16 (user decision,
2026-08-18). After cache + vLLM/parser preflight, run its easy25 baseline. If
it yields at least two valid successes, run the four matched live policy arms:

A. keep recent verbatim;
B. raw action skeleton + verbatim tail;
C. stock Terminus-2 summary;
D. no compaction/truncation.

Arm B has been corrected from the rejected canonical one-liner design. A/B/C
now share a three-policy-compaction cap followed by the same fallback.

## Hard stop rule

No duplicate exp3–21 jobs, GLM cliff runs, full exp6 rerun, or broad model
matrix before the outcome gate. If the Qwen3.8-27B baseline is at floor,
switch model/scaffold; do not manufacture progress with more proxies.
A proper training follow-up is designed as **exp24 GRPO-D** with matched base,
SFT-best and DPO controls (`docs/RL_EXPERIMENT.md`); it starts only after exp22
establishes a non-floor outcome evaluator.

## Submission gate

- on-policy behavioral evidence: **DONE**;
- off-policy-regime warning: **DONE**;
- verifier + oracle validity: **DONE**;
- Qwen3.8-27B cache: **DONE**, snapshot `1d4bf0f...`, 52 GiB;
- full-window serving: **DONE** (77,824 tokens, 51.1 GiB weights, job 803407);
- raw-interface preflights 803407/803548: **retired as invalid proxies**;
- Harbor gate 805028: commands+verifier green but trial timeout; gate fixed;
- partial baseline 805029/data 805030: **cancelled non-results**;
- corrected full Harbor gate 809199: **GREEN**, reward=1, 14 tool calls,
  no exception, real pytest, 19 minutes;
- easy25 baseline 809200: **COMPLETE BUT INFRA-CONFOUNDED**; 6 clean
  successes and 1 clean failure, plus 8 agent timeouts and 10 environment
  start failures. This clears competence but is not a reportable Pass@1;
- full Qwen3.8 data build 809201: **GREEN**, 210 points/12 tasks, task-disjoint
  148 train / 23 validation / 39 test; below the 1,000-train main-run gate;
- immutable pilot snapshot: 66 points/3 tasks (809850), plumbing only;
- exp24 pilot 809859: failed before model load (`packaging`); fixed executor
  module isolation;
- exp24 pilot 813697: executor **GREEN** through `/health`, then failed before
  selector load because clean-node ENV-compress2 lacked `typing_extensions`;
  no training/reward rows/checkpoint. Complete Alliance selector stack is now
  explicit and checked before executor startup. First one-H100 construction
  gates 813707/813711 failed in 10s each before model load: first hierarchical
  StdEnv ordering, then an overwritten Alliance `PYTHONPATH` hid packages from
  `EBPYTHONPREFIXES`. Dependent full-node 813708 never ran. StdEnv is separated,
  module errors are visible, and repo path is now prepended without deleting
  cluster `sitecustomize`. Gate 813716 then loaded the full 4B selector and
  reached GRPOTrainer, exposing TRL 0.29's `reward_func.__name__` requirement;
  fixed and regression-tested. One-H100 gate 813721 is now **GREEN**: full 4B
  load + LoRA + GRPOTrainer construction in 33s. Four-H100 pilot 813724 reached
  step 1/10 but was **CANCELLED AS INVALID**: 8/8 non-JSON raw continuations,
  constant reward=-1, loss/grad=0, no checkpoint. Prepared prompts now use the
  native Qwen3.5 chat template, repeat the JSON contract at the generation
  boundary, and normalize conversational completions. Native-chat gate 828054
  then understood the task but spent all 64 tokens in default `<think>` prose
  (0/4 JSON); it correctly refused release. TRL/preflight now use Qwen3.5's
  native `enable_thinking=False`. One-H100 gate 828065 is now **GREEN**, 4/4
  parseable JSON candidates with diverse selections. Four-H100 integration
  pilot 828066 is **GREEN WITH A FIXED WARNING**: 10/10 steps, 80 rewards,
  28 strict-valid selections, 20 positive rewards, 11/20 nonconstant groups,
  nonzero gradients on all steps, and an 84.97MB adapter. This is plumbing,
  not evidence. Left-padding confirmation 828149 is **GREEN**: 2/2 steps,
  zero padding warnings, all 4 groups nonconstant, nonzero gradients, adapter
  saved. The exp24 plumbing path is now clean; main training remains blocked
  at 148/1,000 task-disjoint train rows. Low-reasoning, concurrency-1 easy25
  replicas r2/r3 (834653/834654) and fixed-port-safe replacement r4 (834686)
  are running. Original r4 job 834655 is invalid and archived. To avoid an
  expected underpowered rebuild, additional independent replicas r5/r6/r7 are
  834739/834740/834741. All six completed; rebuild 834742 produced 1,462
  total task-disjoint rows but only **967 train / 182 validation / 313 test**.
  The 1,000 gate correctly blocked training. Gap replica r8 job 840989 is
  running; rebuild 840990 follows automatically. A CPU integrity gate now
  verifies >=1,000 rows, manifest counts/SHA, chat shape, Qwen3.8 lineage, and
  task disjointness as job 841033 (`afterok:840990`). Powered Dr-GRPO seeds 42
  and 43 are queued as 841034/841035 with `afterok:841033`; if r8 still leaves
  <1,000 rows, the gate fails and both full-node jobs are cancelled untouched;
- exp22 four-arm pilot: **OPEN** after baseline competence check;
- powered outcome row: **OPEN**;
- paper draft/figures: **OPEN**.
