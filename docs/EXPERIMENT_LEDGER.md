# Experiment ledger — paper disposition, not chronology

Last audited: 2026-08-18. **Only on-policy results can support paper claims.**
Off-policy results are retained only for the regime-dependence comparison.
`N` below is usable decision points after the predeclared acting filter.

| Exp | Question | Best valid evidence | ICLR disposition |
|---|---|---|---|
| 1 | Credit decay across compactions | Exact arithmetic: first of four summaries receives 13.5% of last-summary signal | Related-work motivation only |
| 2 | Segment-expanded GRPO weighting | Monte Carlo under stipulated assumptions | Appendix or cut; no GPU work |
| 3 | Is the full-context target length-stable? | On-policy N=12, sampling floor 0.68 | **Inconclusive; cut** |
| 4 | Are tool calls uniquely load-bearing? | Powered on-policy N=25: halt 0.35 vs 0.28, p=0.78 | **Null on-policy**; use only to show off-policy laws fail to transfer |
| 5 | Does wrapper format change behavior? | On-policy N=5 | Superseded by exp21/14; cut |
| 6 | Rate-distortion curves | N=5 complete; N=16 checkpoint after two 24h timeouts, floor 0.45 [0.34,0.56] | **Retire**: floor-dominated, no stable policy ordering, too expensive; exp8/17/23 answer the paper question directly |
| 7 | Does repeated compression compound? | On-policy N=5 | Preliminary; cut from main paper |
| 8 | Does compression preserve the logged real action? | Powered on-policy N=25: full 0.65, keep-recent@25% 0.66, summary 0.46 | **Headline evidence** |
| 9 | Which deployed summary prompt is best? | On-policy N=7 | Preliminary; production baseline moves to exp22 |
| 10 | Does damage persist under continuation? | On-policy N=6, teacher-forced | Preliminary; limitation only |
| 11 | Can D select better summaries? | On-policy selection evidence at 16k (177 examples); fresh-score advantage; earlier p=0.0004 | Secondary result; appendix unless outcome bridge validates D |
| 12 | Does summary ranking transfer between executors? | On-policy N=1 | **Unusable; cut** |
| 13 | Does a deletion manifest help? | Menu confound | **Withdrawn** |
| 14 | Are format preferences model-specific? | Powered Qwen on-policy N=24: native 0.47 vs wrapper 0.48 at top_p=1; no effect. GLM cliff exists off-policy only | **No on-policy Qwen cliff; cut GLM headline unless an on-policy GLM arm is later justified** |
| 15 | What happens at real compaction boundaries? | Construction fixed, then delivery format suppressed acting | Confounded for policy comparison; only keep-recent agreement 0.74 is descriptive |
| 16 / T4 | Can D train a compressor? | 86 on-policy pairs; DPO 0.73 vs base 0.75, p=0.345 | Clean scoped null; appendix, no scale-up before exp22 |
| 17 | How small is the behavioral core? | Powered on-policy N=25: agreement 0.57 at 2%, 0.68 at 25% | **Headline/supporting evidence** |
| 18 | — | No canonical experiment/artifact | Number retired; do not recreate |
| 19 | Does sampled D reflect exact tool TV? | Exact 0.029 vs sampled 0.275, N=10 | Methods validation; appendix |
| 20 | Does verbatim containment predict preservation? | Powered on-policy N=25: rho=-0.23; stronger than NLL (-0.12) | Supporting, predictive-only |
| 21 | Does canonical shorthand preserve action history? | Powered on-policy N=25: wrapped 0.49 vs bare 0.55, p=0.36; raw beats canonical | On-policy null for wrapper; supports **do not rewrite** |
| 22 | Do policies change real task success? | Harness smoke only; offline verifier and oracle reward=1 now green | **SOLE NEXT EXPERIMENT / submission gate** |
| 23 | Raw skeleton+tail vs rewritten one-liners | Powered on-policy N=25: verbatim policies tie (pooled p=0.39); rewriting costs about 13 points | **Headline matched-budget evidence** |
| 24 | Can direct behavioral-reward training work? | Designed only: GRPO-D vs matched base/SFT-best/DPO, >=1k task-split points | **Next major training experiment after exp22 Gate 1; not yet run** |

## The evidence package we already have

The coherent on-policy result is not “skeleton wins.” It is:

1. At a 25% budget, simple verbatim recency is indistinguishable from full
   history on logged-action agreement (exp8).
2. Verbatim policies remain surprisingly useful even at 2% (exp17).
3. Raw skeleton+tail and keep-recent tie, while rewriting the same action
   history into canonical one-liners loses substantially (exp23/21).
4. Two attractive off-policy laws—the block/freeze law and wrapper
   advantage—vanish on our own trajectories (exp4/21). Evaluation regime is
   therefore part of the method, not a nuisance variable.

## Stop list

Until exp22 produces a valid outcome row, do **not** spend GPUs on exp3–21,
DPO scale-up, the full exp6 curve, broad model coverage, or the GLM cliff.
Those jobs can refine a behavioral story but cannot repair its central
external-validity hole.
