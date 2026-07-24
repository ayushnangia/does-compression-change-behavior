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


## Generation-budget and sampling conventions (sourced 2026-07-24, no invented numbers)

Adopted verbatim from the deployment stack, cited:

- **Output cap 10240** = harbor reads `max_output_tokens` from deployment
  model_info (`harbor/llms/lite_llm.py:191`); our TB2 config sets 10240.
  Ours was 768 until 2026-07-24 - an invented cap that could truncate long
  deliberation into a false halt. Requant covers it.
- **Temperature: the literature value is 1.0.** Three independent sources
  agree: CompactionRL (arXiv:2607.05378, experiments section: "top-p=1.0,
  temperature=1.0" in harbor with Terminus-KIRA - the closest published
  setup to ours), harbor's pass-nothing default (`lite_llm.py:310` -> server
  default 1.0), and GLM-4.7's own generation_config (1.0). Our top_p=1.0
  already matches CompactionRL exactly; our temperature=0.7 was invented.
  SEQUENCED SWITCH: the queued requants run at 0.7 to isolate the parser
  fix (one variable at a time); after they land, the default flips to 1.0
  and the headline table reruns under the full cited standard
  (temp 1.0 / top_p 1.0 / max_out 10240 / per-model formats / BFCL match /
  floors + paired permutation).
- **Stop strings**: harbor uses none (runs to EOS/cap). We stop at the first
  closed tool call as a pure compute optimization - measurement-neutral BY
  CONSTRUCTION because our label is the first closed call either way.
- **Halt semantics differ from deployment**: harbor never terminates on
  unparseable output - it auto-fixes truncated JSON, feeds the parse error
  back to the model, and retries the API call 3x (`lite_llm.py:258`,
  parser auto-fixes, error-feedback loop). Our "halt" is a one-shot
  measurement. Deployed, the freeze law manifests as wasted turns and
  retries, not a literal stop. State this in any writeup.

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
