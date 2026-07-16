"""Run the whole thing and print a results table.

For each usable example (a decision point where the full-context model
actually acts), we build each compressed context, sample the model's next
actions, and compare them to the full-context actions. We print, per
compressor: acting rate, how much the behavior changed (vs the noise floor),
and the lookup rate.

    python run_experiment.py --model Qwen/Qwen3.5-4B --num-examples 8
    python run_experiment.py --scaffold        # add the recovery test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from behavior import action_kind, sample_actions
from compressors import TEXT_COMPRESSORS
from data import load_examples, load_examples_file
from metrics import acting_rate, action_change, lookup_rate, normalized_change
from scaffold import recover_actions


def build_context(name, ex, tokenizer, model, device):
    """Return the token ids for one compressor on one example.
    'full' uses the raw context unchanged (the baseline)."""
    if name == "full":
        return ex.context_ids
    old = ex.context_ids[: -len(ex.recent_ids)]
    compressed_old = TEXT_COMPRESSORS[name](old, tokenizer, model, device)
    return compressed_old + ex.recent_ids


def sample_behavior(ctx, old_text, args, model, tokenizer, device, *, seed):
    """Return (actions, used_lookups) for one context.

    With --scaffold, run the recovery loop: offer a tool menu and, when the
    agent looks something up, hand back the relevant slice of the original
    history and let it continue (see scaffold.py). Without it, just sample the
    next action directly. `used_lookups` is None outside scaffold mode.
    """
    if args.scaffold:
        return recover_actions(model, tokenizer, ctx, old_text, device,
                               samples=args.samples, seed=seed)
    acts = sample_actions(model, tokenizer, ctx, device,
                          samples=args.samples, seed=seed)
    return acts, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--num-examples", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--context-tokens", type=int, default=1024)
    ap.add_argument("--recent-tokens", type=int, default=128)
    ap.add_argument("--scaffold", action="store_true",
                    help="offer a tool menu incl. lookup (the recovery test)")
    ap.add_argument("--examples-file", default=None,
                    help="load prefetched examples (see prefetch.py) instead "
                         "of streaming from HuggingFace — needed on offline "
                         "compute nodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    ).eval()
    device = model.device

    if args.examples_file:
        examples = load_examples_file(args.examples_file, args.num_examples)
        print(f"loaded {len(examples)} prefetched examples from {args.examples_file}")
    else:
        examples = load_examples(
            tokenizer, context_tokens=args.context_tokens,
            recent_tokens=args.recent_tokens, num_examples=args.num_examples, seed=args.seed,
        )
    names = ["full", "keep_recent", "summary", "paraphrase", "pointer", "hallucinator"]

    # accumulate per-compressor stats over the usable examples
    stats = {n: {"acting": [], "change": [], "lookup": []} for n in names}
    floors = []
    usable = 0
    t0 = time.time()

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        # the original history that compression throws away; the recovery test
        # serves lookups from this exact text (see scaffold.serve_lookup)
        old_text = tokenizer.decode(
            ex.context_ids[: -len(ex.recent_ids)], skip_special_tokens=True)
        # full-context behavior, twice: once as the reference, once to measure
        # the noise floor (how much identical contexts differ from sampling).
        # In scaffold mode the reference also goes through the menu so the
        # action space matches what the compressed agent is compared against.
        full_a, _ = sample_behavior(ex.context_ids, old_text, args,
                                    model, tokenizer, device, seed=base + 1)
        # only keep examples where the model actually acts with full context,
        # otherwise there is no behavior to compare
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped (full-context model rarely acts)")
            continue
        usable += 1
        full_b, _ = sample_behavior(ex.context_ids, old_text, args,
                                    model, tokenizer, device, seed=base + 2)
        floors.append(action_change(full_a, full_b))

        for name in names:
            ctx = build_context(name, ex, tokenizer, model, device)
            acts, used = sample_behavior(ctx, old_text, args, model, tokenizer,
                                         device, seed=base + 3)
            stats[name]["acting"].append(acting_rate(acts))
            stats[name]["change"].append(action_change(full_a, acts))
            # in scaffold mode: fraction of rollouts that consulted history;
            # otherwise: fraction whose next action was itself a lookup
            look = (sum(used) / len(used)) if used is not None \
                else lookup_rate(acts, action_kind)
            stats[name]["lookup"].append(look)
        print(f"example {ei}: done ({time.time()-t0:.0f}s)")

    if usable == 0:
        print("no usable examples — try a bigger model or more examples")
        return

    floor = sum(floors) / len(floors)
    print(f"\nusable examples: {usable}   sampling floor: {floor:.2f}\n")
    print(f"{'compressor':14s} {'acting':>7s} {'change':>7s} {'vs-floor':>9s} {'lookup':>7s}")
    print("-" * 50)
    rows = {}
    for name in names:
        acting = sum(stats[name]["acting"]) / usable
        change = sum(stats[name]["change"]) / usable
        norm = normalized_change(change, floor)
        look = sum(stats[name]["lookup"]) / usable
        rows[name] = {"acting_rate": acting, "action_change": change,
                      "change_vs_floor": norm, "lookup_rate": look}
        print(f"{name:14s} {acting:7.2f} {change:7.2f} {norm:9.2f} {look:7.2f}")

    Path(args.out).mkdir(exist_ok=True)
    result = {"model": args.model, "usable_examples": usable, "floor": floor,
              "scaffold": args.scaffold, "compressors": rows}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    Path(args.out, f"result_{stamp}.json").write_text(json.dumps(result, indent=2))
    print(f"\nsaved results/result_{stamp}.json")


if __name__ == "__main__":
    main()
