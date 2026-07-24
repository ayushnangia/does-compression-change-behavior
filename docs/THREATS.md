# Threat register: every claim, every attack a reviewer can make

Status codes: DEFENDED (evidence exists) / IN-FLIGHT (job running or queued)
/ UNDEFENDED (no current answer - must fix or must not claim).

## Finding 1: agents freeze when action history is deleted (exp4)

| # | attack | status |
|---|---|---|
| 1.1 | "Halting is your parser failing on formats you didn't handle" | DEFENDED (requant, Jul 24): effect survives at 0.31 vs 0.10 halts (3.1x); old numbers were inflated ~7pts and are retired. AUDIT has the re-quote |
| 1.2 | "Halting is your 768-token cap truncating long deliberation" | DEFENDED: same requant, 10240 budget; differential effect intact |
| 1.3 | "Deleting tool-call blocks leaves structurally mangled turns; the model freezes at malformed text, not missing information" | PARTIALLY DEFENDED: observation-deletion also mangles structure and is free - but the mangling is not identical across conditions. A structure-preserving ablation (replace blocks with placeholder) was never run. UNDEFENDED at the margin |
| 1.4 | "One-shot halt is not deployment: harbor auto-fixes, feeds errors back, retries 3x - your freeze law may vanish under the real loop" | UNDEFENDED except by disclosure; exp22 trajectories will show post-compaction stall rates in the real loop |
| 1.5 | "Two lineages (Qwen, GLM) is not a law" | IN-FLIGHT: H200 plan, 9 lineages |
| 1.6 | "p=0.016 at N=24, and your replications reuse the same data pipeline" | DEFENDED at BH level; powered N=48x3 rerun sequenced after requant |

## Finding 2: containment predicts behavior (exp20)

| # | attack | status |
|---|---|---|
| 2.1 | "N=17, one rate, correlation driven by 6 condition clusters - this is a rank correlation over 6 points wearing 17 points' clothes" | IN-FLIGHT: exp20b (3 rates x N=48, continuous variation) |
| 2.2 | "Containment is confounded with format familiarity: extractive policies preserve BOTH content and surface form - you cannot separate them" | PARTIALLY DEFENDED: exp21 separates them (canonical = information kept, surface form changed) and finds both terms real (format 19pts, canonicalization ~5pts). But that is one model, one domain |
| 2.3 | "Correlation, not causation" | UNDEFENDED as causation; claim must stay predictive ("useful pre-flight check"), never mechanistic |

## Finding 3: format cliffs (exp14, exp21-P3)

| # | attack | status |
|---|---|---|
| 3.1 | "GLM's 0.00 was your parser not knowing GLM's native format" - the arg_key bug is REAL and directly on this number | IN-FLIGHT: requant 66357033. Genuinely at risk. Early log shows real degenerate generation too, so likely partial survival - but the magnitude WILL change |
| 3.2 | "You ran GLM at temperature 0.7; GLM ships 1.0 - the collapse is off-distribution sampling" | UNDEFENDED. The requant also runs at 0.7 (sequencing). A GLM arm at 1.0 is REQUIRED before this claim is published |
| 3.3 | "N=6 examples, 4 samples" (exp14) | UNDEFENDED beyond disclosure; powered rerun needed |
| 3.4 | exp21-P3 (wrapper +19pts, p=0.013): "one model, one domain, N=19" | DEFENDED at BH level; replication on a second model cheap and not yet run |

## Finding 4: tiny core / one-liners (exp17, exp21)

| # | attack | status |
|---|---|---|
| 4.1 | "Your agreement metric = matching the logged action of an agent that scored 0% - you measure fidelity to incompetence" | PARTIALLY DEFENDED: the claim is behavior PRESERVATION (same actions), not quality. But the obvious rejoinder - "preservation of flailing is worthless" - has no answer until competent-model trajectories exist (Trillium) |
| 4.2 | "Agreement is next-action imitation with recency bias; keep-recent is favored by construction, so 'tie' understates nothing - it may overstate keep-recent" | DEFENDED as disclosure (stated in results); exp22 is the real answer |
| 4.3 | "The 2% knee is a property of YOUR trace distribution" | UNDEFENDED beyond one domain |

## Finding 5: select-don't-train (exp11, expB)

| # | attack | status |
|---|---|---|
| 5.1 | "Selection is validated BY the same metric that does the selecting - circular even with fresh-seed rescoring" | UNDEFENDED externally. exp22's D->Pass@1 bridge is the only true fix |
| 5.2 | "The training null is underpowered (350 pairs, 4B, LoRA) - your 'selection works, training doesn't' contrast is rhetoric" | DEFENDED only if phrased as scoped null ("at this dose"); H200 Phase 2 is the real test. NEVER phrase as 'training doesn't work' |

## Finding 6: summaries worst (exp6, exp9, exp15)

| # | attack | status |
|---|---|---|
| 6.1 | "Your summary baselines are strawmen; production compaction is better engineered" | IN-FLIGHT: exp22 arm C is LITERALLY the production Terminus-2 3-step pipeline, unmodified. Until it runs, this attack stands |
| 6.2 | "exp15 was format-confounded by your own admission" | DEFENDED by exclusion (not presented); format-matched redo still owed |

## Cross-cutting attacks (hit every claim at once)

| # | attack | status |
|---|---|---|
| X.1 | "Everything measured at temperature 0.7; the ecosystem standard you yourselves cite is 1.0. Do ANY effects survive deployment sampling?" | UNDEFENDED. The single biggest open technical risk. Sequenced: temp flips after requants; headline reruns at 1.0 are mandatory before submission |
| X.2 | "All results from <=35B models that cannot solve the benchmark" | IN-FLIGHT: Trillium bf16 chain + H200 plan |
| X.3 | "One trace domain (terminal/SWE)" | UNDEFENDED; second domain (e.g. web/tool-API agents) not scheduled. Must be scoped in the title/claims ("coding agents") |
| X.4 | "TV over 8 samples has resolution 1/8; your D values are coarse-grained" | DEFENDED: floors reported, exact-logprob estimator validated (exp19), effects >> resolution |
| X.5 | "Floors differ across experiments; cross-experiment D comparisons are meaningless" | DEFENDED by practice (never compare D across experiments) - but one sentence must say so explicitly |
| X.6 | "You found 4 parser/convention bugs AFTER publishing internal numbers; why believe there is no fifth?" | PARTIALLY DEFENDED: certification harness inverts the authority (in flight); honest answer is 'the audit trail is the evidence of process, and all headline numbers are being requantified' |
| X.7 | "exp2's GRPO simulation assumptions are chosen to maximize the reported bias" | DEFENDED only as "under stated assumptions"; sensitivity sweep over assumptions is cheap and not run |

## What this register implies (the do-not list)

1. Do NOT publish the GLM 0.00 collapse magnitude before requant + a
   temperature-1.0 arm (attacks 3.1, 3.2).
2. Do NOT phrase the training null as "training doesn't work" (5.2).
3. Do NOT claim causation for containment (2.3).
4. Do NOT let any claim rest on agreement-with-logged-actions alone once
   exp22/Trillium numbers exist (4.1).
5. Scope the paper to coding agents explicitly (X.3).
6. The submission gate is: requants harvested + temp-1.0 headline rerun +
   exp20b + exp22 done. Anything earlier is a workshop paper.
