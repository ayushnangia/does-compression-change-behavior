"""EXP 3 (GPU) — Is the full-context behavior target stable across context
lengths?

Claim being tested (raised against our SSL objective): "the full-context
model is also doing lossy inference over a long history ... worth thinking
about whether the full-context target is stable enough across trajectory
lengths to be trusted."

Method: same decision point, same recent history — vary only how much true
prefix precedes it (a length sweep). Sample next-actions at each length and
compare distributions pairwise. The fixed-length sampling floor (two seeded
runs at max length) is the reference: if cross-length drift ~ floor, the
target is length-stable and safe to distill; if drift >> floor, a flat
full-context target imports length artifacts and should be marginalized
over lengths/layouts.

    python exp3_target_stability.py --examples-file ../data/examples_prefetched.json
"""

from __future__ import annotations

import argparse

from common import load_model, save_result, REPO  # noqa: F401  (REPO sets sys.path)
from behavior import sample_actions
from data import load_examples_file
from metrics import acting_rate, action_change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True,
                    help="prefetched examples built at the LARGEST sweep length")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[1024, 2048, 4096])
    ap.add_argument("--num-examples", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)
    lengths = sorted(args.lengths)
    max_len = lengths[-1]

    floors, drift = [], {L: [] for L in lengths[:-1]}
    drift_pad = {L: [] for L in lengths[:-1]}
    pair = {(a, b): [] for i, a in enumerate(lengths) for b in lengths[i + 1:]}
    usable = 0

    for ei, ex in enumerate(examples):
        if len(ex.context_ids) < max_len:
            continue
        base = args.seed * 9973 + ei * 17
        acts = {}
        for L in lengths:
            acts[L] = sample_actions(model, tokenizer, ex.context_ids[-L:],
                                     device, samples=args.samples, seed=base + 1)
        if acting_rate(acts[max_len]) < 0.5:
            print(f"example {ei}: skipped (rarely acts at max length)")
            continue
        usable += 1
        # sampling floor at fixed max length
        ref2 = sample_actions(model, tokenizer, ex.context_ids[-max_len:],
                              device, samples=args.samples, seed=base + 2)
        floors.append(action_change(acts[max_len], ref2))
        # PADDING CONTROL — disentangles the confound: truncation removes real
        # information, so drift(L) mixes 'target instability' with 'the target
        # legitimately differs given less info'. Here we keep the SAME L tokens
        # of true history but pad the prefix with irrelevant trace text from a
        # DIFFERENT example, restoring total length to max_len. Any drift here
        # is positional/length artifact at CONSTANT information.
        filler = examples[(ei + 1) % len(examples)].context_ids
        for L in lengths[:-1]:
            drift[L].append(action_change(acts[max_len], acts[L]))
            padded = filler[: max_len - L] + ex.context_ids[-L:]
            acts_pad = sample_actions(model, tokenizer, padded, device,
                                      samples=args.samples, seed=base + 1)
            drift_pad[L].append(action_change(acts[max_len], acts_pad))
        for (a, b) in pair:
            pair[(a, b)].append(action_change(acts[a], acts[b]))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable: {usable}   sampling floor (fixed {max_len}): {floor:.3f}\n")
    print(f"{'target len':>10s} {'drift(trunc)':>13s} {'drift(padded)':>14s} {'x floor':>8s}")
    rows, rows_pad = {}, {}
    for L in lengths[:-1]:
        d = sum(drift[L]) / len(drift[L])
        dp = sum(drift_pad[L]) / len(drift_pad[L])
        rows[L], rows_pad[L] = d, dp
        print(f"{L:10d} {d:13.3f} {dp:14.3f} {d / floor if floor else float('nan'):8.1f}")
    print("\nreading: drift(padded) ~ floor -> truncation drift was informational"
          "\n         drift(padded) ~ drift(trunc) -> genuine positional instability")

    save_result("exp3_target_stability", {
        "model": args.model, "lengths": lengths, "usable": usable,
        "floor": floor, "drift_vs_max": rows, "drift_padded": rows_pad,
        "pairwise": {f"{a}v{b}": sum(v) / len(v) for (a, b), v in pair.items() if v},
        "raw": {"floors": floors,
                "drift": {str(L): v for L, v in drift.items()},
                "drift_padded": {str(L): v for L, v in drift_pad.items()}},
    })


if __name__ == "__main__":
    main()
