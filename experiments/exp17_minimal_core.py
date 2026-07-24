"""EXP 17 (GPU) — The minimal behavioral core: how little context preserves
behavior?

Motivation (group feedback): if deleting half the tokens costs nothing, where
is the actual floor? We push the kept fraction down to 2% and compare three
strategies, including a pure ACTION SKELETON: the context reduced to only the
tool calls the agent made, nothing else.

Compressors at rates {0.02, 0.05, 0.125, 0.25}:
  keep_recent  recency truncation
  block_aware  drop reasoning, then observations, then random-trim
  skeleton     ONLY the <tool_calls> blocks, concatenated, then random-trim

Metrics per cell: acting rate, coarse divergence (tool-level), agreement with
the logged real action. CIs and raw arrays saved.

    python exp17_minimal_core.py --examples-file ../data/examples_16k_large.json
"""

from __future__ import annotations

import argparse
import random
import re

from common import load_model, save_result, REPO  # noqa: F401
from stats import fmt_ci
from behavior import sample_actions
from data import load_examples_file
from exp4_block_ablation import BLOCK_RES, random_trim
from metrics import acting_rate, action_change_tools

TOOLCALL_BLOCK = re.compile(r"<tool_calls>.*?</tool_calls>", re.DOTALL)


def compress(name, old_ids, old_text, budget, tokenizer, rng):
    ids = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]
    if name == "keep_recent":
        return old_ids[-budget:]
    if name == "block_aware":
        t = BLOCK_RES["drop_reasoning"].sub(lambda m: m.group(1), old_text)
        x = ids(t)
        if len(x) > budget:
            t = BLOCK_RES["drop_observations"].sub(lambda m: m.group(1), t)
            x = ids(t)
        return random_trim(x, budget, rng)
    if name == "skeleton":
        calls = "\n".join(m.group(0) for m in TOOLCALL_BLOCK.finditer(old_text))
        return random_trim(ids(calls), budget, rng)
    raise ValueError(name)


def agreement(actions, logged):
    if logged is None:
        return None
    lt = logged.split("::", 1)[0]
    return sum(a is not None and a.split("::", 1)[0] == lt
               for a in actions) / len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=24)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.02, 0.05, 0.125, 0.25])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)
    comps = ["keep_recent", "block_aware", "skeleton"]

    stats = {(c, r): {"acting": [], "coarse": [], "agree": []}
             for c in comps for r in args.rates}
    floors = []
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        rng = random.Random(base)
        old = ex.context_ids[: -len(ex.recent_ids)]
        old_text = tokenizer.decode(old, skip_special_tokens=True)

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped (full-context model rarely acts)")
            continue
        usable += 1
        full_b = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change_tools(full_a, full_b))

        for c in comps:
            for r in args.rates:
                budget = max(16, int(len(old) * r))
                ids = compress(c, old, old_text, budget, tokenizer, rng)
                acts = sample_actions(model, tokenizer,
                                      list(ids) + list(ex.recent_ids), device,
                                      samples=args.samples, seed=base + 3)
                cell = stats[(c, r)]
                cell["acting"].append(acting_rate(acts))
                cell["coarse"].append(action_change_tools(full_a, acts))
                ag = agreement(acts, ex.logged_action)
                if ag is not None:
                    cell["agree"].append(ag)
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable: {usable}   coarse floor: {floor:.3f}")
    print("\ncells: coarse-D (CI) | acting | agreement\n")
    header = "compressor    " + "".join(f"  R={r:<14g}" for r in args.rates)
    print(header)
    rows = {}
    for c in comps:
        cells = []
        for r in args.rates:
            s = stats[(c, r)]
            n = len(s["acting"])
            co = sum(s["coarse"]) / n
            ac = sum(s["acting"]) / n
            ag = sum(s["agree"]) / len(s["agree"]) if s["agree"] else float("nan")
            cells.append(f"  {co:.2f}|{ac:.2f}|{ag:.2f} ")
            rows[f"{c}@{r:g}"] = {"coarse": co, "acting": ac, "agree": ag}
        print(f"{c:13s}" + "".join(cells))
    for c in comps:
        r = args.rates[0]
        print(f"{c}@{args.rates[0]:g} coarse CI: {fmt_ci(stats[(c, r)]['coarse'])}")

    save_result("exp17_minimal_core", {
        "model": args.model, "usable": usable, "floor": floor,
        "rates": args.rates, "cells": rows,
        "raw": {f"{c}@{r:g}": stats[(c, r)] for c in comps for r in args.rates},
    })


if __name__ == "__main__":
    main()
