"""EXP 4 (GPU) — Do different context blocks carry different behavioral value
per token? (the case against one flat next-action divergence)

Claim being tested: "agent context has structured blocks: tool calls,
observations, files, intermediate plans ... a flat KL can average over very
different kinds of information loss."

Method: block-selective compression at a MATCHED token budget. Each variant
preferentially deletes one block type from the old history (tool calls /
observations / assistant reasoning), then random-deletes further spans until
every variant hits the same budget. A pure random-span control isolates
"tokens removed" from "which tokens removed". If the claim is right, equal
budgets will produce unequal behavior damage AND qualitatively different
failure profiles (halt vs lookup vs different-commit) — which a single flat
divergence would average away.

    python exp4_block_ablation.py --examples-file ../examples_prefetched.json
"""

from __future__ import annotations

import argparse
import random
import re

from common import load_model, save_result, kind_profile, REPO  # noqa: F401
from stats import fmt_ci, paired_permutation_p
from behavior import action_kind, sample_actions
from data import load_examples_file
from metrics import acting_rate, action_change

# ROLE-AWARE block matching. The traces do NOT use <observation> tags: tool
# outputs live in <content> inside role=tool turns, and reasoning in <content>
# inside role=assistant turns. Matching bare tags mislabels blocks (v1 of this
# experiment did: its 'drop_observations' was a no-op and 'drop_reasoning'
# removed reasoning AND observations).
BLOCK_RES = {
    "drop_tool_calls": re.compile(r"<tool_calls>.*?</tool_calls>", re.DOTALL),
    "drop_observations": re.compile(
        r"(<turn index=\d+ role=(?:tool|observation)>)\s*<content>.*?</content>",
        re.DOTALL),
    "drop_reasoning": re.compile(
        r"(<turn index=\d+ role=assistant>)\s*<content>.*?</content>",
        re.DOTALL),
}


def random_trim(ids, budget, rng):
    """Delete random contiguous spans until len(ids) <= budget."""
    ids = list(ids)
    while len(ids) > budget:
        span = min(64, len(ids) - budget)
        start = rng.randrange(0, len(ids) - span)
        del ids[start:start + span]
    return ids


def build_variant(name, old_text, tokenizer, budget, rng):
    """Return token ids for one block-ablation variant at exactly <= budget."""
    if name == "random_control":
        text = old_text
    else:
        rex = BLOCK_RES[name]
        # keep the turn header (group 1) when the pattern captures it
        repl = (lambda m: m.group(1)) if rex.groups else ""
        text = rex.sub(repl, old_text)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return random_trim(ids, budget, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--budget-frac", type=float, default=0.5,
                    help="token budget as a fraction of the old history")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)
    variants = ["random_control", "drop_tool_calls", "drop_observations",
                "drop_reasoning"]

    stats = {v: {"acting": [], "change": [], "profile": []} for v in variants}
    floors = []
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        rng = random.Random(base)
        old_ids = ex.context_ids[: -len(ex.recent_ids)]
        old_text = tokenizer.decode(old_ids, skip_special_tokens=True)
        budget = int(len(old_ids) * args.budget_frac)

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped (full-context model rarely acts)")
            continue
        usable += 1
        full_b = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change(full_a, full_b))

        for v in variants:
            ids = build_variant(v, old_text, tokenizer, budget, rng)
            acts = sample_actions(model, tokenizer, ids + ex.recent_ids, device,
                                  samples=args.samples, seed=base + 3)
            stats[v]["acting"].append(acting_rate(acts))
            stats[v]["change"].append(action_change(full_a, acts))
            stats[v]["profile"].append(kind_profile(acts, action_kind))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable: {usable}   floor: {floor:.3f}   "
          f"budget: {args.budget_frac:.0%} of old history\n")
    print(f"{'variant':18s} {'acting (95% CI)':>18s} {'change (95% CI)':>18s} "
          f"{'none':>6s} {'lookup':>7s} {'commit':>7s}")
    rows = {}
    for v in variants:
        n = len(stats[v]["acting"])
        acting = sum(stats[v]["acting"]) / n
        change = sum(stats[v]["change"]) / n
        prof = {k: sum(p[k] for p in stats[v]["profile"]) / n
                for k in ("none", "lookup", "commit")}
        rows[v] = {"acting": acting, "change": change, **prof}
        print(f"{v:18s} {fmt_ci(stats[v]['acting']):>18s} "
              f"{fmt_ci(stats[v]['change']):>18s} "
              f"{prof['none']:6.2f} {prof['lookup']:7.2f} {prof['commit']:7.2f}")
    # primary endpoint: paired test of each block variant vs random control
    for v in variants[1:]:
        p = paired_permutation_p(stats[v]["change"], stats["random_control"]["change"])
        print(f"paired p (change, {v} vs random_control): {p:.3f}")

    save_result("exp4_block_ablation", {
        "model": args.model, "usable": usable, "floor": floor,
        "budget_frac": args.budget_frac, "variants": rows,
        "raw": {v: {"acting": stats[v]["acting"], "change": stats[v]["change"]}
                for v in variants},
    })


if __name__ == "__main__":
    main()
