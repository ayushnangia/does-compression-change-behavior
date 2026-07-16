"""EXP 10 (GPU) — Does compression damage HEAL or PERSIST as the real
trajectory continues?

exp7 measured pure re-compression decay (summary of summary). This measures
the realistic case: compress ONCE, then teacher-force the trace's ACTUAL
continuation (real actions + observations from the original run) and
re-measure behavioral divergence at each subsequent decision point.

  D(j) falls toward the floor  -> fresh grounded interaction washes the
                                  damage out (compaction is self-healing;
                                  single-step D overstates harm)
  D(j) stays flat / grows      -> damage persists past the compaction
                                  boundary (single-step D understates harm,
                                  and early summaries deserve strong credit)

Both full and compressed branches receive IDENTICAL continuations, so any
divergence at step j is attributable to the single compression at step 0.

    python exp10_propagation.py --examples-file ../examples_64.json
"""

from __future__ import annotations

import argparse

from common import load_model, save_result, REPO  # noqa: F401
from behavior import sample_actions
from compressors import TEXT_COMPRESSORS
from data import load_examples_file
from metrics import acting_rate, action_change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True,
                    help="must be prefetched WITH future segments")
    ap.add_argument("--compressor", default="summary",
                    choices=list(TEXT_COMPRESSORS))
    ap.add_argument("--num-examples", type=int, default=24)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = [ex for ex in load_examples_file(args.examples_file, None)
                if len(ex.future_segments) >= args.steps][: args.num_examples]
    print(f"{len(examples)} examples with >= {args.steps} future segments")

    stats = {j: {"change": [], "acting": []} for j in range(args.steps + 1)}
    floors = {j: [] for j in range(args.steps + 1)}
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        old = ex.context_ids[: -len(ex.recent_ids)]

        full_a0 = sample_actions(model, tokenizer, ex.context_ids, device,
                                 samples=args.samples, seed=base + 1)
        if acting_rate(full_a0) < 0.5:
            print(f"example {ei}: skipped (full-context model rarely acts)")
            continue
        usable += 1

        comp = TEXT_COMPRESSORS[args.compressor](old, tokenizer, model, device)
        full_ctx = list(ex.context_ids)
        comp_ctx = comp + list(ex.recent_ids)

        for j in range(args.steps + 1):
            if j > 0:   # teacher-force the trace's real next segment into BOTH
                seg = ex.future_segments[j - 1]
                full_ctx = full_ctx + seg
                comp_ctx = comp_ctx + seg
            fa = sample_actions(model, tokenizer, full_ctx, device,
                                samples=args.samples, seed=base + 10 + j)
            fb = sample_actions(model, tokenizer, full_ctx, device,
                                samples=args.samples, seed=base + 40 + j)
            ca = sample_actions(model, tokenizer, comp_ctx, device,
                                samples=args.samples, seed=base + 70 + j)
            floors[j].append(action_change(fa, fb))
            stats[j]["change"].append(action_change(fa, ca))
            stats[j]["acting"].append(acting_rate(ca))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    print(f"\nusable: {usable}   compressor: {args.compressor}\n")
    print(f"{'step j':>6s} {'change':>7s} {'floor':>6s} {'x floor':>8s} {'acting':>7s}")
    rows = {}
    for j in range(args.steps + 1):
        n = len(stats[j]["change"])
        ch = sum(stats[j]["change"]) / n
        fl = sum(floors[j]) / n
        ac = sum(stats[j]["acting"]) / n
        rows[j] = {"change": ch, "floor": fl, "acting": ac}
        print(f"{j:6d} {ch:7.2f} {fl:6.2f} "
              f"{ch / fl if fl else float('nan'):8.1f} {ac:7.2f}")

    save_result("exp10_propagation", {
        "model": args.model, "usable": usable,
        "compressor": args.compressor, "steps": rows,
    })


if __name__ == "__main__":
    main()
