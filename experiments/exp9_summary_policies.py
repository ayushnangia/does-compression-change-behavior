"""EXP 9 (GPU) — Audit deployed compaction policies AND search prompt space.

Two readings of one table:
  AUDIT  — what do real products' compaction prompts do to behavior? We
           implement prompts modeled on deployed systems (Claude Code
           /compact-style structured handoff, OpenHands condenser-style)
           next to our naive baseline.
  SEARCH — the SSL thesis in miniature: if behavioral distortion D is a good
           objective, we can OPTIMIZE prompts against it without any
           training. The block-aware prompt is exp4's finding turned into
           an instruction; if it wins, measurement -> design works.

All policies get the same budget and are scored on: acting, change (vs
floor), entropy, and grounded agreement with the trace's logged action.

    python exp9_summary_policies.py --examples-file ../examples_64.json
"""

from __future__ import annotations

import argparse

from common import load_model, save_result, REPO  # noqa: F401
from behavior import sample_actions
from compressors import _generate, _ids
from data import load_examples_file
from metrics import acting_rate, action_change, action_entropy

POLICIES = {
    # our naive baseline (the repo's current summary prompt)
    "naive": (
        "[Summarize the agent trace above so the work can be continued. "
        "Keep file paths, commands, errors, and decisions. Be concise.]"),
    # modeled on Claude Code /compact: structured handoff
    "structured_handoff": (
        "[Compact the conversation above into a handoff note with sections: "
        "1) Task; 2) Files examined/modified (exact paths); 3) Commands run "
        "and their key results; 4) Errors encountered and their status; "
        "5) Current state; 6) Next steps. Be precise and complete.]"),
    # modeled on OpenHands condenser: keep actions, condense observations
    "condense_observations": (
        "[Condense the trace above: keep every action the agent took "
        "(commands, edits) as-is, but compress tool outputs and observations "
        "to their key facts only.]"),
    # exp4's measured finding turned into an instruction
    "block_aware": (
        "[Compress the trace above. COPY all tool calls, commands, and file "
        "paths VERBATIM. Drop or heavily compress reasoning and commentary. "
        "Keep error messages exactly.]"),
    # CONTROL: empty summary — isolates the cost of the compaction-note
    # format discontinuity itself (wrapper + mid-turn recent fragment).
    # Any policy's damage should be read relative to this, not to zero.
    "wrapper_only": None,
    # ablation extremes
    "minimal": "[Briefly summarize the work above.]",
    "next_step_only": (
        "[State only: what must be done next, and the specific facts (paths, "
        "names, errors) needed to do it.]"),
}


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
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)

    stats = {p: {"acting": [], "change": [], "entropy": [], "agree": [],
                 "length": []}   # actual token length: policies differ under
             for p in POLICIES}  # the same cap -> report the rate confound
    floors = []
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
        full_b = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change(full_a, full_b))

        for p, instr in POLICIES.items():
            if instr is None:          # wrapper_only control: no summary text
                text, gen = "", []
            else:
                prompt = old + _ids(tokenizer, f"\n\n{instr}\nSummary:")
                gen = _generate(model, tokenizer, prompt, args.budget, device)
                text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            stats[p]["length"].append(len(gen))
            wrapped = (f"<turn index=0 role=user>\n<content>\n[Summary of "
                       f"work so far: {text}]\n</content>\n</turn>\n")
            ctx = _ids(tokenizer, wrapped) + ex.recent_ids
            acts = sample_actions(model, tokenizer, ctx, device,
                                  samples=args.samples, seed=base + 3)
            stats[p]["acting"].append(acting_rate(acts))
            stats[p]["change"].append(action_change(full_a, acts))
            stats[p]["entropy"].append(action_entropy(acts))
            ag = agreement(acts, ex.logged_action)
            if ag is not None:
                stats[p]["agree"].append(ag)
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable: {usable}   floor: {floor:.3f}   budget: {args.budget} tok\n")
    print(f"{'policy':22s} {'acting':>7s} {'change':>7s} {'entropy':>8s} "
          f"{'agree':>6s} {'len':>5s}")
    rows = {}
    for p in POLICIES:
        n = len(stats[p]["acting"])
        row = {
            "acting": sum(stats[p]["acting"]) / n,
            "change": sum(stats[p]["change"]) / n,
            "entropy": sum(stats[p]["entropy"]) / n,
            "agree": (sum(stats[p]["agree"]) / len(stats[p]["agree"])
                      if stats[p]["agree"] else float("nan")),
            "length": sum(stats[p]["length"]) / n,
        }
        rows[p] = row
        print(f"{p:22s} {row['acting']:7.2f} {row['change']:7.2f} "
              f"{row['entropy']:8.2f} {row['agree']:6.2f} {row['length']:5.0f}")

    save_result("exp9_summary_policies", {
        "model": args.model, "usable": usable, "floor": floor,
        "budget": args.budget, "policies": rows,
    })


if __name__ == "__main__":
    main()
