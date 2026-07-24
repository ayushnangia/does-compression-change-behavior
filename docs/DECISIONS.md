# Decision provenance and externalization map

Who chose each measurement convention, why, what external standard it
follows, and what would replace it with something citable. Written so that
disagreement with a convention lands on the stated rationale, not on a
person.

**Authorship note.** The measurement conventions below were designed by the
coding agent (Claude, Anthropic) during development sessions, implemented
under test, and reviewed through results by Ayush Nangia and the group. The
research questions and experiment goals came from the group; the mechanical
conventions did not. Where a convention has upstream lineage it is cited;
where it is homegrown it is marked HOMEGROWN with its replacement path.

## The conventions

| convention | origin | rationale | external anchor / replacement path |
|---|---|---|---|
| Action = first tool call in the continuation | HOMEGROWN | agents act through tool calls; the first one is what would execute | OpenAI tool-calling spec defines the object; **replace parsing with vLLM's structured `tool_calls` field** (below) |
| Label = `name::args60` | HOMEGROWN | comparable string, args capped to avoid free-text noise | BFCL (Berkeley Function-Calling Leaderboard) compares calls by AST match on name+args - adopting their matcher would externalize this |
| Halt = no tool call in any accepted format | HOMEGROWN | distinguishes "acts differently" from "stops acting"; parser totality tested over 4 formats | with vLLM structured output, halt = empty `tool_calls`, i.e. the serving layer's verdict, not ours |
| `<think>` stripped before parsing | follows model cards | Qwen3/GLM document think-blocks as non-executable deliberation | model-card semantics; vLLM reasoning-parser does this server-side |
| 3 granularities (tool / verb / exact) | HOMEGROWN | one number hides whether changes are cosmetic or substantive | BFCL AST match approximates "exact"; verb-level is ours alone and is labeled as such in results |
| D = total variation over 8-sample action distributions | standard | TV is the standard probability distance; policy-distance framing follows TRPO (Schulman 2015); see RELATED.md lineage | textbook; the 8-sample estimator is validated against exact logprobs in exp19 |
| Floor-referencing (subtract full-vs-full resample) | standard practice | sampling noise must not be reported as effect | exp19 shows it is load-bearing (floor approx 0.275 at tool level) |
| Paired permutation tests across examples | standard | small N, no distributional assumptions | textbook |
| Logged-action agreement as grounding | HOMEGROWN (forced) | quantized models score 0% on TB2, so task success was unavailable | replaced by real TB2 Pass@1 the moment the bf16 Trillium run lands |
| Trajectory format for on-policy data | external | we consume harbor's `trajectory.json` as-is | harbor / Terminal-Bench 2.0 schema |

## The externalization plan (ordered by value)

1. **Parsing -> vLLM, PER-MODEL parsers.** Serve with
   `--enable-auto-tool-choice --tool-call-parser <model's own>`:
   `qwen3_engine` for Qwen3.5 (its native format is XML-parameter,
   `<function=name><parameter=k>v</parameter>`, NOT hermes JSON - verified
   against the model's chat template), `glm47_moe` for GLM-4.7. Never a
   generic parser for a specific model.
   HISTORY NOTE (2026-07-24): our regex parser did not accept Qwen3.5's
   native XML format until today - it parsed as halt. Fixed (parser now
   total over 5 formats, tested), and a quantification job measures how
   often models actually emitted that format in our regime; verdict in
   AUDIT.md when it lands. This incident is the argument for this entire
   document.
2. **Call comparison -> BFCL AST matching.** Their matcher is the de facto
   standard for "is this the same function call". Adopt for the exact
   granularity; keep tool-level as the coarse view.
3. **Grounding -> Terminal-Bench Pass@1.** Already in motion (Trillium
   chain); retires the weakest homegrown convention entirely.
4. **Trace schema -> harbor.** Already done; we never invented a trajectory
   format.

Until 1-2 are executed, published numbers rest on the homegrown parser -
which is why it is property-tested (parser totality, tests 1-20) and why
its behavior is documented in behavior.py docstrings rather than implied.
