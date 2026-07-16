"""EXP 6 (GPU) — The rate-distortion curve. The core object of the framing.

First principles: if compression is a rate-distortion problem where the
distortion is behavioral change, then the fundamental measurement is
D(R) — behavior change as a function of token budget — per compressor.
Everything else (which blocks to keep, which summarizer, which format) is
a question about which compressor's curve sits lowest.

Four compressors, same budgets, same examples:
  keep_recent  — truncation (the trivial baseline)
  random       — random span deletion (the uninformed baseline)
  block_aware  — delete reasoning prose first, keep tool calls (exp4's
                 lesson turned into a compressor), then random-trim
  summary      — model-written summary of that token budget

Distortion is action_change vs the full-context behavior, read against the
sampling floor. acting_rate is reported alongside (halting is the failure
mode flat distortion hides).

    python exp6_rate_distortion.py --examples-file ../examples_prefetched.json
"""

from __future__ import annotations

import argparse
import random
import re

from common import load_model, save_result, REPO  # noqa: F401
from behavior import sample_actions
from compressors import SUMMARIZE, _generate, _ids
from data import load_examples_file
from metrics import acting_rate, action_change, action_change_tools

# role-aware: reasoning = content inside assistant turns; observations =
# content inside tool turns (the traces do NOT use <observation> tags)
REASONING_RE = re.compile(
    r"(<turn index=\d+ role=assistant>)\s*<content>.*?</content>", re.DOTALL)
OBSERVATION_RE = re.compile(
    r"(<turn index=\d+ role=(?:tool|observation)>)\s*<content>.*?</content>", re.DOTALL)


def random_trim(ids, budget, rng):
    ids = list(ids)
    while len(ids) > budget:
        span = min(64, len(ids) - budget)
        start = rng.randrange(0, len(ids) - span)
        del ids[start:start + span]
    return ids


def compress(name, old_ids, old_text, budget, tokenizer, model, device, rng):
    """Return old-history replacement of <= budget tokens."""
    if name == "keep_recent":
        return old_ids[-budget:]
    if name == "random":
        return random_trim(old_ids, budget, rng)
    if name == "block_aware":
        # hierarchical: drop reasoning first, then observations if still over
        # budget, keep tool calls to the end (exp4's measured ordering)
        text = REASONING_RE.sub(lambda m: m.group(1), old_text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            text = OBSERVATION_RE.sub(lambda m: m.group(1), text)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return random_trim(ids, budget, rng)
    if name == "summary":
        wrap_overhead = 32   # leave room for the wrapper so it isn't cut
        prompt = old_ids + _ids(tokenizer, SUMMARIZE)
        gen = _generate(model, tokenizer, prompt,
                        max(16, budget - wrap_overhead), device)
        text = tokenizer.decode(gen, skip_special_tokens=True)
        wrapped = (f"<turn index=0 role=user>\n<content>\n[Summary of work so "
                   f"far: {text}]\n</content>\n</turn>\n")
        return _ids(tokenizer, wrapped)[: budget]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.125, 0.25, 0.5, 0.75])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)
    compressors = ["keep_recent", "random", "block_aware", "summary"]

    stats = {(c, r): {"acting": [], "change": [], "coarse": []}
             for c in compressors for r in args.rates}
    floors = []
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        rng = random.Random(base)
        old_ids = ex.context_ids[: -len(ex.recent_ids)]
        old_text = tokenizer.decode(old_ids, skip_special_tokens=True)

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped (full-context model rarely acts)")
            continue
        usable += 1
        full_b = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change(full_a, full_b))

        for c in compressors:
            for r in args.rates:
                budget = max(16, int(len(old_ids) * r))
                ids = compress(c, old_ids, old_text, budget,
                               tokenizer, model, device, rng)
                acts = sample_actions(model, tokenizer, ids + ex.recent_ids,
                                      device, samples=args.samples, seed=base + 3)
                stats[(c, r)]["acting"].append(acting_rate(acts))
                stats[(c, r)]["change"].append(action_change(full_a, acts))
                stats[(c, r)]["coarse"].append(action_change_tools(full_a, acts))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable: {usable}   floor: {floor:.3f}")
    print("\nD(R): behavior change (and acting rate) by kept fraction\n")
    header = "compressor    " + "".join(f"  R={r:<11g}" for r in args.rates)
    print(header)
    rows = {}
    for c in compressors:
        cells = []
        for r in args.rates:
            n = len(stats[(c, r)]["change"])
            ch = sum(stats[(c, r)]["change"]) / n
            ac = sum(stats[(c, r)]["acting"]) / n
            co = sum(stats[(c, r)]["coarse"]) / n
            cells.append(f"  {ch:.2f}/{co:.2f} ({ac:.2f}) ")
            rows[f"{c}@{r}"] = {"change": ch, "acting": ac, "coarse": co}
        print(f"{c:13s}" + "".join(cells))
    print("\ncell = change (acting). Lower change at equal R = better curve.")

    save_result("exp6_rate_distortion", {
        "model": args.model, "usable": usable, "floor": floor,
        "rates": args.rates, "curves": rows,
        "raw": {f"{c}@{r:g}": {"change": stats[(c, r)]["change"],
                                     "coarse": stats[(c, r)]["coarse"],
                                     "acting": stats[(c, r)]["acting"]}
                for c in compressors for r in args.rates},
        "raw_floors": floors,
    })


if __name__ == "__main__":
    main()
