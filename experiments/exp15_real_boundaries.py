"""EXP 15 (GPU) — Score PRODUCTION compactions at genuine window-exhaustion
boundaries.

Our TB2 trajectories contain real compaction events: Terminus hit its context
limit, wrote a handoff summary, and continued. For each event we reconstruct:

  reference   — the uncompacted context (raw pre-compaction history spliced
                back in place of the marker+handoff)
  production  — the context exactly as Terminus left it (its real handoff)
  block_aware — our measured design rule applied to the same history
  keep_recent — truncation at matched budget
  wrapper_only— the format-seam control

and measure behavioral divergence from the reference plus agreement with the
logged next action. "How good are the summaries a production agent actually
writes, against a measured design and a dumb baseline?" — at the exact
moments compaction actually happens.

    python exp15_real_boundaries.py   (paths default to our TB2 jobs)
"""

from __future__ import annotations

import argparse
import glob
import json
import re

from common import load_model, save_result, REPO  # noqa: F401
from stats import fmt_ci, paired_permutation_p
from behavior import parse_action, sample_actions
from data import _serialize
from exp4_block_ablation import BLOCK_RES, random_trim
from metrics import acting_rate, action_change

MARKER = "Performed context summarization"
WRAP = ("<turn index=0 role=user>\n<content>\n[Summary of work so far: "
        "{s}]\n</content>\n</turn>\n")


def find_events(path):
    """Yield (msgs, marker_idx, handoff_text) for each compaction event."""
    d = json.load(open(path))
    steps = d.get("steps", [])
    msgs = [{"role": s.get("source", "event"), "content": s.get("message"),
             "tool_calls": s.get("tool_calls"),
             "observation": s.get("observation")} for s in steps]
    for i, s in enumerate(steps):
        if s.get("source") == "system" and MARKER in str(s.get("message", "")):
            handoff = str(steps[i + 1].get("message", "")) if i + 1 < len(steps) else ""
            yield msgs, i, handoff


def build_variants(msgs, mi, handoff, tokenizer, model, device, budget, rng):
    """Return (ref_ids, {name: ids}, logged) or None if unusable."""
    task = {"instance_id": "t", "repo": "t"}
    pre = _serialize({**task, "trajectory": msgs[:mi]})          # true history
    # post-compaction segment: skip marker + handoff message (mi, mi+1)
    post = _serialize({**task, "trajectory": msgs[mi + 2:]})
    m = re.search(r"<tool_calls>", post)
    if not m:
        return None
    post_head = post[: m.start()]                                # up to next real action
    logged = parse_action(post[m.start(): m.start() + 2000])

    ids = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]
    pre_ids = ids(pre)
    post_ids = ids(post_head)
    if len(pre_ids) < 2 * budget or len(post_ids) > 6000:
        return None
    ref = (pre_ids + post_ids)[-16384:]   # uncompacted reference (16k cap:
    # 24k x 8 samples OOMs A100-40 under HF generate; 16k matches the program)

    va = {}
    va["production"] = ids(WRAP.format(s=handoff.strip()[:6000])) + post_ids
    ba = BLOCK_RES["drop_reasoning"].sub(lambda x: x.group(1), pre)
    ba = BLOCK_RES["drop_observations"].sub(lambda x: x.group(1), ba)
    va["block_aware"] = random_trim(ids(ba), budget, rng) + post_ids
    va["keep_recent"] = pre_ids[-budget:] + post_ids
    va["wrapper_only"] = ids(WRAP.format(s="")) + post_ids
    return ref, va, logged


def agreement(actions, logged):
    if logged is None:
        return None
    lt = logged.split("::", 1)[0]
    return sum(a is not None and a.split("::", 1)[0] == lt
               for a in actions) / len(actions)


def main():
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--traj-glob",
                    default="/scratch/anangia/tb2/jobs/tb2-qwen35-*/*/agent/trajectory.json")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--max-events", type=int, default=23)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    names = ["production", "block_aware", "keep_recent", "wrapper_only"]
    stats = {n: {"change": [], "acting": [], "agree": []} for n in names}
    floors = []
    usable = 0

    events = []
    for p in sorted(glob.glob(args.traj_glob)):
        events.extend(find_events(p))
    print(f"{len(events)} real compaction events found")

    for ei, (msgs, mi, handoff) in enumerate(events[: args.max_events]):
        base = args.seed * 9973 + ei * 17
        built = build_variants(msgs, mi, handoff, tokenizer, model, device,
                               args.budget, random.Random(base))
        if built is None:
            print(f"event {ei}: skipped (unusable geometry)")
            continue
        ref, variants, logged = built
        full_a = sample_actions(model, tokenizer, ref, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"event {ei}: skipped (reference rarely acts)")
            continue
        usable += 1
        full_b = sample_actions(model, tokenizer, ref, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change(full_a, full_b))
        for n in names:
            acts = sample_actions(model, tokenizer, variants[n], device,
                                  samples=args.samples, seed=base + 3)
            stats[n]["change"].append(action_change(full_a, acts))
            stats[n]["acting"].append(acting_rate(acts))
            ag = agreement(acts, logged)
            if ag is not None:
                stats[n]["agree"].append(ag)
        print(f"event {ei}: done")

    if not usable:
        print("no usable events")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable events: {usable}   floor: {floor:.3f}   "
          f"budget: {args.budget} tok\n")
    print(f"{'variant':14s} {'change (CI)':>16s} {'acting (CI)':>16s} {'agree':>6s}")
    rows = {}
    for n in names:
        ag = stats[n]["agree"]
        rows[n] = {"change": sum(stats[n]["change"]) / len(stats[n]["change"]),
                   "acting": sum(stats[n]["acting"]) / len(stats[n]["acting"]),
                   "agree": sum(ag) / len(ag) if ag else float("nan")}
        print(f"{n:14s} {fmt_ci(stats[n]['change']):>16s} "
              f"{fmt_ci(stats[n]['acting']):>16s} {rows[n]['agree']:6.2f}")
    if usable > 1:
        p = paired_permutation_p(stats["block_aware"]["change"],
                                 stats["production"]["change"])
        print(f"\npaired p (change, block_aware vs production): {p:.3f}")

    save_result("exp15_real_boundaries", {
        "model": args.model, "usable": usable, "floor": floor,
        "budget": args.budget, "variants": rows,
        "raw": {n: stats[n] for n in names},
    })


if __name__ == "__main__":
    main()
