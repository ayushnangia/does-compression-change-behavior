# Does compressing an agent's memory change what it does?

When an AI agent works on a long task, its history of steps grows until it no
longer fits in the model's context window. Every real system handles this the
same way: it **compresses** the old history (usually by summarizing it) and
keeps going.

This repo measures one thing: **after you compress the history, does the agent
still behave the same way?** Not "does it remember the facts" — does it take
the same *next action*.

## The question, precisely

Cut a real agent trace at a decision point (right before it calls a tool).
Ask the model "what do you do next?" two ways:

1. with the **full** history, and
2. with a **compressed** history.

Then compare the two sets of next actions. If compression is harmless, they
match. If it isn't, they diverge — or the agent stops acting entirely.

## What we measure (three numbers, all plain)

- **acting rate** — of the model's sampled continuations, what fraction are an
  actual tool call (vs. drifting into prose and doing nothing). Full context
  is the baseline.
- **action change** — how different the compressed agent's next actions are
  from the full-context agent's, on a 0–1 scale. We report it against a
  **floor** (how much two runs of the *same* full context differ, just from
  random sampling) so you can tell real change from noise.
- **lookup rate** — when the agent is offered a menu of tools, how often does
  it choose to *re-read* something instead of committing to a task action.
  This is the key control: a different action might be the agent smartly
  going to look something up, not damage. (See "the recovery test" below.)

## The compressors we compare

Each takes the old history and shrinks it; the most recent few turns are
always kept raw (that's how real systems work).

| name | what it does | why it's here |
|---|---|---|
| `full` | nothing (baseline) | the thing everything is compared to |
| `keep_recent` | keep only the last half of the tokens | simplest lossy method |
| `summary` | model summarizes the old history | what real products actually do |
| `paraphrase` | reword the old history, keep all the info | control: separates "lost info" from "different words" |
| `pointer` | drop the old history but leave "see file X" pointers | control: damage the agent *can* recover from |
| `hallucinator` | drop the old history, say "all done, proceed confidently" | control: damage the agent *cannot* recover from |

The last two are the honest test of the metric: a good measure must say
`pointer` (recoverable) is *better* than `hallucinator` (not recoverable).

## The recovery test (why a "different action" isn't automatically bad)

A person with a compressed memory doesn't fail — they look something up. An
agent can too: re-read a file, grep the code. So a different next action might
be **adaptive** (going to re-read), not **damage**. To measure this honestly,
`run_experiment.py --scaffold` (implemented in `scaffold.py`) offers the agent
an explicit tool menu including lookups, and when it looks something up we hand
back the relevant slice of the *original, pre-compression* history and let it
continue — up to two lookups before it must act. If the compressed agent
recovers and reaches the same action, that compression was fine; if it stays
silent even with the menu, the information loss was real. In `--scaffold`
mode the full-context reference goes through the same menu, so the two are
compared over the same action space, and the `lookup` column reports the
fraction of rollouts that consulted history.

## What we found (in the bigger study this repo distills)

- Summaries don't make the agent act *wrong* so much as make it **stop
  acting** — the acting rate collapses even though the compressed history
  still nominally lets the agent continue.
- Standard quality checks (perplexity / "can it answer questions") say
  compression is nearly lossless while behavior falls apart. The two come
  apart.
- The honest open question, which the scaffold/recovery test targets: is the
  halt real information loss, or is the agent about to go look something up?

None of these are settled by a single run — the point of this repo is that you
can run it, see it, and extend it.

## Run it

```bash
pip install -r requirements.txt

# small model, runs on a free Colab GPU or slowly on CPU:
python run_experiment.py --model Qwen/Qwen3.5-4B --num-examples 8

# add the recovery test (offer a tool menu, allow lookups):
python run_experiment.py --model Qwen/Qwen3.5-4B --scaffold

# on a cloud GPU with the full-size model (needs Modal):
modal run modal_run.py
```

Results print as a table and save to `results/`.

## How it's organized (each file does one thing)

- `data.py` — load real agent traces and cut them at decision points.
- `compressors.py` — the seven ways to shrink the history.
- `behavior.py` — sample the model's next actions and parse them.
- `scaffold.py` — the recovery test: tool menu + hand back history on lookup.
- `metrics.py` — acting rate, action change, the noise floor.
- `run_experiment.py` — tie it together, print the table.
- `modal_run.py` — run on a cloud GPU.

## Add your own experiment

- New compression method → add a function to `compressors.py` (one function,
  takes old + recent context, returns compressed context).
- New way to measure behavior → add to `metrics.py`.
- The whole loop is in `run_experiment.py` and is ~80 readable lines.

## Honesty notes

- Traces come from `nvidia/Open-SWE-Traces` — real software-agent runs, but
  written by a *different* model than the one you measure with. That's fine
  for the behavior comparison (same model vs itself) but means the "quiz"
  score is scored on another model's text.
- A small model will show weaker, noisier effects than the 8B used in the full
  study. Turn up `--num-examples` and `--samples` for cleaner numbers.
- Decide your thresholds ("what counts as a real change") *before* you look at
  the results. It's the difference between measuring and fooling yourself.
