"""EXP 5 (GPU) — How much of compaction behavior is the INTERFACE rather than
the information? (inference-level proxy for the "operating inside the
compaction loop" claim)

Claim being tested: "CompactionRL is trained for compaction-enabled
execution, and the gains don't transfer to single-window evaluation. so part
of the gain may come from learning how to operate inside the compaction
loop, rather than learning a better compressed representation."

Method: hold the summary CONTENT fixed (one greedy summary per example) and
vary only the FORMAT in which it is presented — inline note, paper-style
resume template, assistant's own voice, bare text. If behavior shifts
materially across formats at identical information content, then a
meaningful share of compaction performance lives in interface familiarity,
not representation quality — which is exactly the component RL-in-the-loop
would learn and exactly the component that would NOT transfer to
single-window evaluation.

Honest scope: the full claim is about training; this measures the
inference-time footprint of the mechanism (formats the base model was never
adapted to). A positive result makes the claim plausible, not proven.

    python exp5_format_vs_content.py --examples-file ../examples_prefetched.json
"""

from __future__ import annotations

import argparse

from common import load_model, save_result, kind_profile, REPO  # noqa: F401
from behavior import action_kind, sample_actions
from compressors import SUMMARIZE, _generate, _ids
from data import load_examples_file
from metrics import acting_rate, action_change

FORMATS = {
    "inline_note": (
        "<turn index=0 role=user>\n<content>\n[Summary of work so far: {s}]\n"
        "</content>\n</turn>\n"),
    "resume_template": (
        "<turn index=0 role=user>\n<content>\nEarlier interaction history was "
        "compacted to stay within the context window. Summary of prior work:\n"
        "{s}\nResume the task from this summary and the recent turns below.\n"
        "</content>\n</turn>\n"),
    "assistant_voice": (
        "<turn index=0 role=assistant>\n<content>\nI have compacted my notes "
        "on the work so far: {s}\nI will now continue from here.\n"
        "</content>\n</turn>\n"),
    "bare": "{s}\n",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--summary-budget", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)

    stats = {f: {"acting": [], "change": [], "profile": []} for f in FORMATS}
    floors = []
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        old_ids = ex.context_ids[: -len(ex.recent_ids)]

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped (full-context model rarely acts)")
            continue
        usable += 1
        full_b = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change(full_a, full_b))

        # ONE summary per example (greedy) -> content constant across formats
        prompt = old_ids + _ids(tokenizer, SUMMARIZE)
        gen = _generate(model, tokenizer, prompt, args.summary_budget, device)
        summary = tokenizer.decode(gen, skip_special_tokens=True).strip()

        for fmt, template in FORMATS.items():
            ids = _ids(tokenizer, template.format(s=summary))
            acts = sample_actions(model, tokenizer, ids + ex.recent_ids, device,
                                  samples=args.samples, seed=base + 3)
            stats[fmt]["acting"].append(acting_rate(acts))
            stats[fmt]["change"].append(action_change(full_a, acts))
            stats[fmt]["profile"].append(kind_profile(acts, action_kind))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable: {usable}   floor: {floor:.3f}   "
          "(identical summary content in every row)\n")
    print(f"{'format':16s} {'acting':>7s} {'change':>7s} "
          f"{'none':>6s} {'lookup':>7s} {'commit':>7s}")
    rows = {}
    for fmt in FORMATS:
        n = len(stats[fmt]["acting"])
        acting = sum(stats[fmt]["acting"]) / n
        change = sum(stats[fmt]["change"]) / n
        prof = {k: sum(p[k] for p in stats[fmt]["profile"]) / n
                for k in ("none", "lookup", "commit")}
        rows[fmt] = {"acting": acting, "change": change, **prof}
        print(f"{fmt:16s} {acting:7.2f} {change:7.2f} "
              f"{prof['none']:6.2f} {prof['lookup']:7.2f} {prof['commit']:7.2f}")

    save_result("exp5_format_vs_content", {
        "model": args.model, "usable": usable, "floor": floor,
        "formats": rows,
    })


if __name__ == "__main__":
    main()
