"""EXP 7 (GPU) — Does behavioral distortion COMPOUND across repeated
compactions?

Claims being grounded: (a) "as the compaction chain gets deeper, earlier
summaries receive a much weaker learning signal, while the final few
compactions are much more directly responsible" — the credit-assignment
argument presumes early compactions matter because their damage propagates;
(b) CompactionRL caps rollouts at 3 compactions — is that where the chain
breaks?

Method: iterated summarization. k=1 summarizes the true old history; each
further step summarizes the PREVIOUS summary (same token budget), always
keeping the recent turns raw — the pure re-compression bottleneck, isolated.
We report behavior change D(k), acting rate, and action entropy vs the
full-context reference for k = 1..max_chain.

Honest scope: real compaction chains interleave NEW interaction between
compactions; iterated summarization isolates the re-compression decay
component (a lower bound on chain damage). If D(k) grows with k, early
summary quality gates everything downstream — the behavioral basis for
arguing summaries deserve uniform (not temporally discounted) credit. If
D(k) is flat after k=1, the chain-depth concern is overstated.

    python exp7_compaction_chain.py --examples-file ../examples_prefetched.json
"""

from __future__ import annotations

import argparse

from common import load_model, save_result, REPO  # noqa: F401
from behavior import sample_actions
from compressors import SUMMARIZE, _generate, _ids
from data import load_examples_file
from metrics import acting_rate, action_change, action_entropy


def summarize(ids, budget, tokenizer, model, device):
    """One compaction step: model-written summary of `ids`, wrapped the way
    the repo's summary compressor wraps it (a compaction note turn)."""
    prompt = list(ids) + _ids(tokenizer, SUMMARIZE)
    gen = _generate(model, tokenizer, prompt, budget, device)
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    wrapped = (f"<turn index=0 role=user>\n<content>\n[Summary of work so "
               f"far: {text}]\n</content>\n</turn>\n")
    return _ids(tokenizer, wrapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--summary-budget", type=int, default=256)
    ap.add_argument("--max-chain", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)

    ks = list(range(1, args.max_chain + 1))
    stats = {k: {"acting": [], "change": [], "entropy": []} for k in ks}
    floors, full_entropies = [], []
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
        full_entropies.append(action_entropy(full_a))

        state = old_ids
        for k in ks:
            state = summarize(state, args.summary_budget, tokenizer, model, device)
            acts = sample_actions(model, tokenizer, state + ex.recent_ids,
                                  device, samples=args.samples, seed=base + 3)
            stats[k]["acting"].append(acting_rate(acts))
            stats[k]["change"].append(action_change(full_a, acts))
            stats[k]["entropy"].append(action_entropy(acts))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    h_full = sum(full_entropies) / len(full_entropies)
    print(f"\nusable: {usable}   floor: {floor:.3f}   "
          f"full-context entropy: {h_full:.2f} bits\n")
    print(f"{'chain k':>7s} {'change':>7s} {'x floor':>8s} "
          f"{'acting':>7s} {'entropy':>8s}")
    rows = {}
    for k in ks:
        n = len(stats[k]["change"])
        ch = sum(stats[k]["change"]) / n
        ac = sum(stats[k]["acting"]) / n
        en = sum(stats[k]["entropy"]) / n
        rows[k] = {"change": ch, "acting": ac, "entropy": en}
        print(f"{k:7d} {ch:7.2f} {ch / floor if floor else float('nan'):8.1f} "
              f"{ac:7.2f} {en:8.2f}")

    save_result("exp7_compaction_chain", {
        "model": args.model, "usable": usable, "floor": floor,
        "full_entropy": h_full, "summary_budget": args.summary_budget,
        "chain": rows,
    })


if __name__ == "__main__":
    main()
