# Plan: scale the behavioral-compression program to the real compaction regime

Decisions (user, all defaults): 16k pilot then 64k · Narval-first, migrate
only on OOM · 200 DPO pairs auto-continuing to 1000 on held-out win · the 23
real compaction events are measurement-only for now.

## Evidence base (ledger)
- E1: 129 on-policy trajectories (median ~21k tok, max 228k); **23 contain
  genuine window-exhaustion compaction events with Terminus's actual
  summaries** (trajectory.json scan).
- E2: vLLM 0.24 offline `LLM` class imports in ENV-vllm; our
  metrics/parser/data-file modules run there with zero added deps;
  Qwen3.5-9B native context 262,144 with mostly linear-attention layers
  (long-context KV is cheap); def-zhijing fairshare recovered to 0.41.

## must_haves
truths:
- vLLM scorer produces D within the sampling floor of the HF scorer on
  identical examples (distributional equivalence, not token equality)
- >=5x measured throughput vs HF path on the same workload
- 16k results reported with CIs against 4k results (same metrics, same floor
  protocol)
- DPO held-out split is BY TASK/REPO (not index)
artifacts:
- experiments/vllm_scorer.py (+ equivalence test), examples_16k.json,
  exp15 real-boundary results JSON, scaled exp4/exp6/exp8 results,
  round-1 adapter + held-out eval JSON
key_links:
- scorer consumed by exp11 pair generation (not a parallel fork of logic)
- real-compaction examples carry Terminus's actual summary text into exp15

## Implementation DAG

### T1 — vLLM batched scorer (gating; everything downstream consumes it)
Objective: drop-in `sample_actions`/`sample_texts` equivalents backed by an
in-process vllm.LLM engine, batching all candidate contexts per example.
Files: `experiments/vllm_scorer.py` (new), `tests/test_scorer_equivalence*`
(new), no changes to behavior.py (HF path stays as reference + rollback).
Env: ENV-vllm (verified imports). Invariants: same seeds protocol
(per-example bases), same max_new/temperature/top_p, parse_action shared.
Test: same 8 examples through both paths -> |D_hf - D_vllm| <= floor;
acting-rate CIs overlap. Throughput logged.
Risks: vLLM sampling RNG differs from HF (expected; equivalence is
distributional); qwen3_5 support verified for serving, assumed for offline
class (same engine) — probe first in the job.
DoD: equivalence test passes + >=5x throughput measured.

### T2 — Long-context on-policy prefetch (parallel with T1)
Objective: examples_16k.json from ALL TB2 trajectories at
--context-tokens 16384; NEW: detect real compaction events (system steps
matching 'Performed context summarization') and store
`compaction_event: true` + the actual Terminus summary text alongside.
Files: prefetch_onpolicy.py (extend), data.py (optional field, default
None — old files stay loadable).
DoD: >=48 examples at 16k with logged actions; >=15 real-boundary examples
with captured production summaries. Login node, no GPU.

### T3 — 16k pilot measurement (after T1+T2)
Objective: do the 4k laws survive the real regime?
- exp4 (block law), exp6 (D(R)), exp8 (grounding) at 16k, N=32, via scorer.
- **exp15 (new): the real-boundary study** — at the 23 genuine compaction
  events, score D and grounded agreement of: (a) the summary Terminus
  ACTUALLY wrote, (b) block_aware, (c) keep_recent, (d) wrapper_only, at
  matched budget. "We scored a production agent's real compactions."
Files: exp15_real_boundaries.py (new), scaled job script.
DoD: tables + CIs; explicit 4k-vs-16k comparison section; exp15 figure.

### T4 — Scaled pairs + iterative DPO (after T1; pairs parallel with T3)
Objective: 200 gap-filtered pairs at 16k via scorer, with the marginalized
target (average reference over 2 context lengths — exp3v2 lesson); DPO
round 1 (existing 2-GPU boring-path config); held-out eval SPLIT BY TASK.
Gate (Q3 default): adapter beats base on held-out D AND grounded agreement
-> auto-generate 1000 pairs, round 2. Else: documented negative + SFT-on-best
fallback consideration.
Files: exp11 (--scorer vllm flag + marginalized target + task-split),
dpo_job (unchanged), evaluate_compressor (task-split arg).
DoD: held-out verdict either way, with CIs.

### T5 — 64k probe (after T3)
Objective: single-example 64k scoring feasibility on A100-40 (linear
attention should hold; OOM -> Q2 default: port to H100 cluster then).
Then D(R) + exp8 at 64k, N=16.
DoD: feasibility verdict + (if feasible) 64k tables.

## Validation matrix
| Claim at scale | Check |
|---|---|
| scorer faithful | equivalence test vs HF (T1) |
| block law at 16k | exp4@16k vs 4k, CIs (T3) |
| truncation>summary at 16k | exp6@16k (T3) |
| D grounded at 16k | exp8@16k (T3) |
| production summaries rank | exp15 vs alternatives (T3) |
| D trains compressors at scale | T4 held-out, task-split |

## Rejected alternatives
- HTTP client -> vLLM server (simpler but ~2-3x, loses batching control)
- Liger/exotic memory flags for DPO (failed 3x; boring 2-GPU path is proven)
- Immediate 64k (KV risk unquantified; 16k pilot de-risks)

## Rollback / failure handling
HF scorer path untouched -> any T1 failure reverts to current throughput.
T3/T4 jobs are additive (new results files). T5 OOM has a named migration
path. No existing experiment code is modified except additive flags.

## Residual unknowns
vLLM offline-class behavior for qwen3_5 (probe is T1 step 1); whether 23
real events give enough power for exp15 (report CIs, frame as case study
if underpowered); Narval queue variance.
