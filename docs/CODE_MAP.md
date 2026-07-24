# Code map: what each file does, in plain language

Read this before reading the code. For who chose each convention and what
is homegrown vs external: docs/DECISIONS.md.

## The measurement, end to end (one sentence per step)

1. `data.py` loads an example: a long agent history (token ids) split into
   OLD (compressible) and RECENT (never touched).
2. `compressors.py` shrinks the OLD part by some policy.
3. `behavior.py` asks the model "what do you do next?" 8 times from
   (compressed OLD + RECENT) and 8 times from the full history, and parses
   each answer into an action label like `execute_bash::{"command": "ls"}`.
4. `metrics.py` compares the two sets of 8 labels: how different is the
   behavior (D), how often did it act at all (acting rate) - always against
   the noise floor of sampling the SAME full context twice.
5. `experiments/stats.py` says whether differences are real (paired tests,
   scipy-backed).

## File by file

| file | what it is | homegrown or external? |
|---|---|---|
| `behavior.py` | sampling + the action parser (5 formats) | parser is homegrown BUT format specs come from each model's chat template; being replaced by vLLM per-model parsers (DECISIONS plan #1) |
| `compressors.py` | the compression policies under test | homegrown by design - these are the experimental conditions |
| `metrics.py` | D at 3 granularities, acting/halt rates | TV distance is textbook; BFCL AST matching for exact-equality |
| `data.py` | example loading/saving | trivial |
| `scaffold.py` | **LEGACY: a synthetic 5-tool environment** (read_file/grep/run_tests/edit/submit) used only by early experiments (exp3, exp5) | fully homegrown, and yes hardcoded - kept for reproducibility of those two experiments only. Everything since exp8 uses REAL agent traces (harbor/Terminus format), not this menu |
| `experiments/exp*.py` | one experiment each; the docstring at the top of each file states design, conditions, and (from exp11 on) pre-registered predictions | - |
| `experiments/common.py` | model loading, result saving | - |
| `experiments/vllm_scorer.py` | same sampling via vLLM (6x faster), equivalence-tested against the HF path | - |
| `exp22/` (branch `exp22-outcome`) | compaction policies inside LIVE harbor episodes | actions parsed by HARBOR'S OWN parser, imported as-is; agents subclass Terminus-2 overriding one method |
| `tb2/` | offline Terminal-Bench harness | harbor + vLLM, config-only |
| `tests/run_tests.py` | 73 checks, run before every job | - |

## The honest answer to "are tools/actions hardcoded?"

Three different situations, often confused:

1. **`scaffold.py`'s 5-tool menu: yes, hardcoded, and legacy.** It was the
   first synthetic environment (exp3/exp5). No current experiment uses it.
   It stays only so those early results remain reproducible.
2. **The action labels in `behavior.py`: not hardcoded, but homegrown.**
   The parser reads whatever tool call the model emits (any tool name, any
   args) in 5 formats specified by the models' own chat templates. The
   remaining homegrown part is the label NORMALIZATION, and the migration
   path to vLLM's per-model parsers is DECISIONS plan #1.
3. **exp22 (the outcome experiment): taken from harbor as-is.** Actions are
   whatever Terminus-2 defines (keystrokes + task_complete), parsed by
   harbor's own parser including its auto-fixes. This is the direction all
   of it moves: the scaffold defines the action space, we measure on top.

## Reading order for a new contributor

`README.md` -> this file -> `behavior.py` (150 lines, the heart) ->
`experiments/exp4_block_ablation.py` (the strongest finding, typical
experiment shape) -> `docs/DECISIONS.md` -> `docs/AUDIT.md`.
