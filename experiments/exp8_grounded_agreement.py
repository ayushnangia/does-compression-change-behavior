"""EXP 8 (GPU) — Ground the distortion measure against REALITY, not just the
model's own full-context behavior.

The critique all self-referential metrics face: "you're comparing the model
to itself; the full-context reference may be wrong too." But our traces
contain the action the ORIGINAL agent actually took at each decision point
(it sits right after the cut). That is an external reference.

Method: for each compressor/budget, measure agreement between sampled next
actions and the logged action, at two granularities:
  tool  — same tool name (coarse intent)
  exact — same label (tool + first 60 chars of arguments)
If compression lowers agreement-with-reality in step with our behavior-change
metric D, then D is grounded; if D moves but grounded agreement does not,
D is measuring something else.

    python exp8_grounded_agreement.py --examples-file ../data/examples_64.json
"""

from __future__ import annotations

import argparse


from common import load_model, save_result, REPO  # noqa: F401
from behavior import sample_actions
from compressors import TEXT_COMPRESSORS
from data import load_examples_file
from metrics import acting_rate, action_change


def agreement(actions, logged):
    """(tool-level, exact-level) agreement of sampled actions with the
    logged action."""
    if logged is None:
        return None
    lt = logged.split("::", 1)[0]
    tool = sum(a is not None and a.split("::", 1)[0] == lt for a in actions)
    exact = sum(a == logged for a in actions)
    return tool / len(actions), exact / len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True,
                    help="must be prefetched WITH logged actions")
    ap.add_argument("--num-examples", type=int, default=32)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = [ex for ex in load_examples_file(args.examples_file, None)
                if ex.logged_action][: args.num_examples]
    print(f"{len(examples)} examples with logged actions")
    names = ["full", "keep_recent", "summary", "pointer", "hallucinator"]

    stats = {n: {"tool": [], "exact": [], "acting": [], "change": []}
             for n in names}
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        old = ex.context_ids[: -len(ex.recent_ids)]

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped (full-context model rarely acts)")
            continue
        usable += 1

        for n in names:
            if n == "full":
                ctx = ex.context_ids
            else:
                comp = TEXT_COMPRESSORS[n](old, tokenizer, model, device)
                ctx = comp + ex.recent_ids
            acts = full_a if n == "full" else sample_actions(
                model, tokenizer, ctx, device, samples=args.samples, seed=base + 3)
            ag = agreement(acts, ex.logged_action)
            stats[n]["tool"].append(ag[0])
            stats[n]["exact"].append(ag[1])
            stats[n]["acting"].append(acting_rate(acts))
            stats[n]["change"].append(action_change(full_a, acts))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    print(f"\nusable: {usable}   (agreement = with the ORIGINAL agent's "
          "logged action)\n")
    print(f"{'compressor':14s} {'tool-agree':>10s} {'exact':>7s} "
          f"{'acting':>7s} {'change':>7s}")
    rows = {}
    for n in names:
        k = len(stats[n]["tool"])
        row = {m: sum(stats[n][m]) / k for m in ("tool", "exact", "acting", "change")}
        rows[n] = row
        print(f"{n:14s} {row['tool']:10.2f} {row['exact']:7.2f} "
              f"{row['acting']:7.2f} {row['change']:7.2f}")

    save_result("exp8_grounded_agreement", {
        "model": args.model, "usable": usable, "compressors": rows,
    })


if __name__ == "__main__":
    main()
