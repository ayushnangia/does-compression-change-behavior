# Model coverage plan: 8x H200 (1,128GB) + OpenRouter

**Precision rule (non-negotiable): release precision only.** bf16 locally, or
the vendor's native release format (gpt-oss ships MXFP4 - no bf16 exists).
Anything that cannot be hosted in full precision goes to OpenRouter with the
provider and precision pinned and verified. Zero community-quantized weights
anywhere in the paper (the int4-scores-0% lesson, tb2/README.md).

**Experiment tiers** (disclosed, not hidden):
- **Tier F (full suite)**: D at 3 granularities, block law (exp4), format
  phenotype (exp14), one-liners (exp21), minimal core (exp17). Models <= 2 GPUs.
- **Tier S (survey)**: exp4 + exp14 probes at N=6, plus TB2 Pass@1. The TP=8
  giants; a full suite there would burn the allocation for one model.
- **Tier A (API)**: TB2 Pass@1 + trajectory collection + format probes via
  chat. No raw-token D suite (needs local weights; docs/DECISIONS.md).

## Local roster (all bf16 / native release; dates are HF createdAt)

| model | released | params | bf16 | GPUs | tier | covers |
|---|---|---|---|---|---|---|
| Qwen3.5-4B / 9B / 27B | 2025 | 4-27B | 8-56GB | 1 | F | scale axis, dense, hybrid-linear-attn; 9B = incumbent measuring model |
| Qwen3.5-35B-A3B | 2025 | 35B | 70GB | 1 | F | MoE anchor; TB2 ~27% class |
| **Qwen-AgentWorld-35B-A3B** | 2026-06-22 | 35B | 69GB | 1 | F | **agent-tuned matched pair vs vanilla 35B-A3B** |
| GLM-4.7-Flash | 2026 | 30B-A3B | 60GB | 1 | F | lineage 2 (Zhipu) |
| GLM-4.5-Air | 2025 | 106B | 212GB | 2 | F | lineage 2 at scale |
| **gpt-oss-120b** | 2025-08 | 120B | 63GB native MXFP4 | 1 | F | lineage 3 (OpenAI) |
| **Leanstral-1.5-119B-A6B** | 2026-07-01 | 119B | 238GB | 2 | F | lineage 4 (Mistral), sparse-active A6B |
| **Laguna-XS-2.1** | 2026-06-20 | 33B | 67GB | 1 | F | lineage 5 (Poolside) - THE code-agent lab |
| **Laguna-S-2.1** | 2026-07-13 | 118B | 235GB | 2 | F | Poolside flagship-small; second matched pair (XS vs S scale) |
| **Nemotron-3-Puzzle-75B-A9B** | 2026-06-24 | 75B | 151GB | 2 | F | lineage 6 (NVIDIA), puzzle-distilled MoE |
| **Hy3** | 2026-07-02 | 299B | 598GB | TP=8 | S | lineage 7 (Tencent) flagship |
| **MiniMax-M3** | 2026-06-02 | 427B | 854GB | TP=8 | S | lineage 8 (MiniMax); largest bf16 we can host |
| **Trinity-Large-Thinking** | 2026-04-01 | 399B | 797GB | TP=8 | S | lineage 9 (Arcee), thinking-mode flagship |
| Qwen3.5-397B-A17B | 2025 | 403B | 807GB | TP=8 | S | Qwen flagship; up-scale oracle within lineage 1 |

## OpenRouter roster (Tier A; provider + precision pinned before any run)

| model | released | params | why API |
|---|---|---|---|
| **GLM-5.2** | 2026-06-16 | 753B | 1.5TB bf16 - cannot host |
| **Kimi-K2.7-Code** | 2026-06-11 | 1,059B | 2.1TB |
| **Nemotron-3-Ultra-550B-A55B** | 2026-06-03 | 561B | 1.12TB = weights alone eat the cluster |
| 1-2 closed frontier | - | - | grounding reference row only |

**Closed-frontier candidates for the reference rows** (API-only, no open
weights on HF as of 2026-07-24 - verified by search, community repos with
these names are finetunes/merges): Kimi-K3, GPT-5.5, Sol, Terra, Fable.
Include whichever have OpenRouter endpoints at run time; for each, pin the
provider, record the endpoint + date (closed models change silently behind
fixed names - trajectories must carry the run date), and run the one-task
smoke before the full spend. These rows ground the paper's numbers against
the frontier; they join no local experiment.

## The reviewer rebuttal this buys

- **9 independent lineages** (Alibaba, Zhipu, OpenAI, Mistral, Poolside,
  NVIDIA, Tencent, MiniMax, Arcee) + 2 more via API (Moonshot) = 11 labs
- **Scale 4B -> 753B** (190x), dense / MoE / sparse-active / hybrid-linear-attn
- **Two matched pairs**: agent-tuned vs general (AgentWorld vs 35B-A3B, same
  arch and size) and within-lab scale (Laguna XS vs S)
- **12 of 18 models released within 8 weeks** of the experiments
- Release precision throughout; tiers disclosed per model

## Per-model pre-flight gate (docs/DECISIONS.md; mechanical, no exceptions)

1. Pull chat template -> confirm native tool-call format
2. Confirm a dedicated vLLM parser exists IN OUR INSTALLED VERSION
   (biggest real risk for Hy3 / M3 / Trinity / Laguna - check before
   burning a GPU-day; upgrade vLLM or drop the model, never hand-parse)
3. `python tests/run_tests.py` (73 checks)
4. One smoke generation consumed on the target GPUs
5. Eval params: temp 1.0 / top_p 1.0 / max_out 10240 (cited standard)

## Hour budget (8 GPUs x ~14 days = ~2,700 GPU-hours)

| block | GPU-hours |
|---|---|
| Tier F suite x 11 models (approx 60 GPUh each incl. TB2) | ~700 |
| Tier S probes + TB2 x 4 giants (TP=8, approx 40h wall each) | ~1,300 |
| Phase 2 training (full-FT 9B student, 2k pairs) + headline rerun at cited standard | ~400 |
| One-liner long-memory test + 64k arms + slack for reruns | ~300 |

Priorities if the window shrinks: AgentWorld pair -> gpt-oss-120b ->
Laguna pair -> Leanstral -> Nemotron-75B -> giants in TB2-score order.
