# ICLR outline — the single source of narrative truth

Title: **Does Compression Change Behavior? Measuring What Context
Compaction Does to Coding Agents**

Thesis (one sentence): context compression for agents is not text
compression — what matters is not the information you keep but the
behavior you preserve, and behavior lives in surprising places (the
agent's own actions, the wrapper format, a 2% core).

Rules for this file: every claim cites its experiment + the AUDIT-approved
number. STATUS says what blocks it. Nothing goes in the paper that is
BLOCKED here. Quotable numbers come from docs/AUDIT.md requant sections
only.

## Section skeleton -> experiments -> figures

| § | section | experiments | quotable numbers (AUDIT) | STATUS |
|---|---------|-------------|--------------------------|--------|
| 1 | Intro (freeze law as Fig 1) | exp4 | halts 0.31 vs 0.10 (3.1x) @0.7; 0.42 vs 0.21 (2.0x) @1.0 | READY; structure-preserving ablation strengthens (Wave 1) |
| 2 | Related work | — | — | BLOCKED: citation-graph walk (paper/LITERATURE.md) |
| 3 | The instrument: D, floors, theory | exp19, exp8, stats.py | exact 0.029 vs sampled 0.275; grounding 0.67–0.73 | READY; add PDL/AIS framing paragraph |
| 4.1 | Law 1: freeze (block value) | exp4 requant + std | as §1; observations free (0.09 vs 0.10; 0.17 @1.0) | READY (with 1.3 ablation caveat until run) |
| 4.2 | Law 2: containment | exp20, exp20b | rho −0.38/−0.44/−0.45 (N=23 x3); NLL +0.22..+0.30 | READY (predictive-only phrasing, threat 2.3) |
| 4.3 | Law 3: format cliff + wrapper | exp14, exp21, GLM requant | GLM 0.00 vs 0.75 (N=24); wrapper +18pts p=0.023 @0.7; +12 p=0.121 @1.0 | BLOCKED: GLM temp-1.0 arm (threat 3.2); powered wrapper N~64 @1.0 (Wave 1) |
| 4.4 | Law 4: tiny core | exp17, exp21 | 0.59 agreement @330 tokens (2%); ~870-token canonical history 0.53–0.61 vs 0.68 raw | READY (one model caveat, threat 4.3) |
| 5 | Selection vs training | exp11, T4/expB | selection p=0.0004; fresh 0.612 vs 0.790; 16k: 0.67 vs 0.72 non-overlap CIs; DPO null 0.73 vs 0.75 p=0.345 | READY as scoped null ("at this dose"); E-B repo-split fix owed (Wave 0) |
| 6 | The outcome bridge | exp22 (+ TB2 bf16 traces) | — | BLOCKED: H100 pilot (Wave 2), full matrix H200 |
| 7 | Limitations | — | one domain (scoped in title); ≤35B local; D@0 with exp10 context | READY |
| A | Appendix: audit | docs/AUDIT.md near-verbatim | 4 parser bugs found, all headlines requantified, none died | READY — this is a feature, present it as one |

## Figure plan

1. **Fig 1 (the hook):** halt-rate bars by deleted block type, control floor
   line, both temps. "Agents freeze when you delete their own actions."
2. **Fig 2:** containment-vs-coarse-D scatter, 3 rates overlaid, NLL inset.
3. **Fig 3:** format cliff — GLM native vs wrapper acting rate; wrapper
   effect at 2% budget (content held constant).
4. **Fig 4:** agreement vs budget (the 2% knee), compressor families as
   curves; canonical-shorthand saturation point marked.
5. **Fig 5:** exp22 bridge — per-arm Pass@1 vs offline D at compaction
   events. (Placeholder until Wave 2 pilot.)

## What is CUT from the paper (and why — do not resurrect)

- exp13 (manifest masked by menu confound), exp15 v1/v2 (format-confounded,
  withdrawn), exp3 v1 (truncation confound), exp6 label-level (ceiling),
  exp7 GLM arm (measured format effect), anything floor>0.5 at label
  granularity. Full list: COAUTHOR.md §11.
- exp1/exp2 (CompactionRL analyses): one paragraph in related work as
  motivation ("deployed credit schemes sit at a known-bad point of the
  TD(λ) family"), NOT a section. They are a different paper's contribution.

## Do-not list (from THREATS.md, paper-shaped)

1. No GLM cliff magnitude before the temp-1.0 arm.
2. Wrapper effect: "significant at 0.7, directional at 1.0" until powered run.
3. "Training showed no effect at this dose" — never "training doesn't work".
4. Containment is predictive, never causal/mechanistic.
5. Never compare D across experiments; say so once, explicitly.
6. Scope every claim sentence to coding agents.

## Submission gate (unchanged from THREATS §7)

requants harvested ✅ + exp20b ✅ + temp-1.0 headline ✅ + GLM temp-1.0 ❌
+ powered wrapper @1.0 ❌ + exp22 (pilot minimum) ❌. Anything earlier is a
workshop paper.
