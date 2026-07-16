"""EXP 11 / E-A (GPU) — Is behavioral distortion D a usable REWARD signal?
Best-of-N summary selection, plus DPO pair generation.

The thesis in miniature, zero training: at each compaction point, sample N
candidate summaries, score each by immediate behavioral distortion D, and
SELECT the argmin. If the selected summary also wins on (a) grounded
agreement with the trace's logged action and (b) teacher-forced downstream
behavior, then D is a valid dense reward for compaction — the signal that
dissolves the credit-assignment problem (see PLAN.md).

Also writes every scored candidate to a fixed-name pairs file
(summary_pairs.jsonl) for DPO training (E-B): chosen = argmin-D,
rejected = argmax-D per example (first NUM examples are train; the rest of
the examples file is held out for evaluation).

    python exp11_best_of_n.py --examples-file ../examples_64.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_model, save_result, REPO  # noqa: F401
from stats import fmt_ci, paired_permutation_p
from behavior import sample_actions, sample_texts
from compressors import SUMMARIZE, _ids
from data import load_examples_file
from metrics import acting_rate, action_change

WRAP = ("<turn index=0 role=user>\n<content>\n[Summary of work so far: "
        "{s}]\n</content>\n</turn>\n")


def agreement(actions, logged):
    if logged is None:
        return None
    lt = logged.split("::", 1)[0]
    return sum(a is not None and a.split("::", 1)[0] == lt
               for a in actions) / len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B",
                    help="EXECUTOR: behavior reference + scoring")
    ap.add_argument("--summarizer-model", default=None,
                    help="who WRITES candidate summaries. For DPO pair "
                         "generation this MUST be the trainee (e.g. 4B): "
                         "round-1 failed because 9B-written pairs are "
                         "off-policy for the 4B (pref-accuracy < chance)")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=24)
    ap.add_argument("--n-summaries", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs-out", default="results/summary_pairs.jsonl")
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    if args.summarizer_model and args.summarizer_model != args.model:
        stok, smodel, sdev = load_model(args.summarizer_model)
        assert stok("probe <tool_calls>")["input_ids"] == \
            tokenizer("probe <tool_calls>")["input_ids"], "tokenizer mismatch"
    else:
        smodel, sdev = model, device
    examples = load_examples_file(args.examples_file, args.num_examples)

    best_d, rand_d, best_ag, rand_ag = [], [], [], []
    down_best, down_rand = [], []
    floors = []
    pairs = []
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        old = ex.context_ids[: -len(ex.recent_ids)]
        old_text = tokenizer.decode(old, skip_special_tokens=True)

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped (full-context model rarely acts)")
            continue
        usable += 1
        full_b = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change(full_a, full_b))

        # N candidate summaries (temperature-sampled -> diverse), written by
        # the SUMMARIZER model (= the DPO trainee for on-policy pairs)
        prompt = old + _ids(tokenizer, SUMMARIZE)
        cands = sample_texts(smodel, tokenizer, prompt, sdev,
                             samples=args.n_summaries, max_new=args.budget,
                             temperature=0.9, seed=base + 5)

        scored = []
        for si, text in enumerate(cands):
            ctx = _ids(tokenizer, WRAP.format(s=text.strip())) + ex.recent_ids
            acts = sample_actions(model, tokenizer, ctx, device,
                                  samples=args.samples, seed=base + 3)
            scored.append({
                "summary": text.strip(),
                "d": action_change(full_a, acts),
                "acting": acting_rate(acts),
                "agree": agreement(acts, ex.logged_action),
                "ctx": ctx,
            })

        scored.sort(key=lambda r: r["d"])
        best, worst = scored[0], scored[-1]
        mean_d = sum(r["d"] for r in scored) / len(scored)
        # WINNER'S CURSE GUARD: the selected summary must be re-scored with
        # FRESH samples — reporting its selection-time D would inflate the
        # best-of-N gap by construction (selection on noisy scores).
        fresh = sample_actions(model, tokenizer, best["ctx"], device,
                               samples=args.samples, seed=base + 21)
        best_d.append(action_change(full_a, fresh))
        rand_d.append(mean_d)          # expected D of a random pick (no selection -> unbiased)
        fresh_ag = agreement(fresh, ex.logged_action)
        if fresh_ag is not None:
            best_ag.append(fresh_ag)
            rand_ag.append(sum(r["agree"] for r in scored) / len(scored))

        # teacher-forced downstream: does the D-selected summary also behave
        # better at the NEXT real decision point?
        if ex.future_segments:
            seg = ex.future_segments[0]
            f_next = sample_actions(model, tokenizer, ex.context_ids + seg,
                                    device, samples=args.samples, seed=base + 8)
            med = scored[len(scored) // 2]
            for row, sink in ((best, down_best), (med, down_rand)):
                acts = sample_actions(model, tokenizer, row["ctx"] + seg,
                                      device, samples=args.samples, seed=base + 9)
                sink.append(action_change(f_next, acts))

        pairs.append({
            "prompt": old_text[-8000:] + "\n\n" + SUMMARIZE.strip(),
            "chosen": best["summary"], "rejected": worst["summary"],
            "d_chosen": best["d"], "d_rejected": worst["d"],
        })
        print(f"example {ei}: D best {best['d']:.2f} vs mean {mean_d:.2f}")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\nusable: {usable}   floor: {floor:.3f}   N={args.n_summaries}\n")
    print(f"{'':22s} {'select-by-D':>18s} {'random pick':>18s}")
    print(f"{'immediate D':22s} {fmt_ci(best_d):>18s} {fmt_ci(rand_d):>18s}")
    print(f"{'grounded agreement':22s} {fmt_ci(best_ag):>18s} {fmt_ci(rand_ag):>18s}")
    print(f"{'downstream D (step+1)':22s} {fmt_ci(down_best):>18s} {fmt_ci(down_rand):>18s}")
    # PRIMARY ENDPOINT (pre-registered): downstream D, select-by-D vs random.
    # immediate D is confirmatory only (selection and outcome share machinery).
    if len(down_best) > 1:
        p = paired_permutation_p(down_best, down_rand)
        print(f"\npaired p (downstream D, best vs median): {p:.3f}")
    if len(best_d) > 1:
        print(f"paired p (immediate D, fresh-scored):     "
              f"{paired_permutation_p(best_d, rand_d):.3f}")

    out = Path(args.pairs_out)
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nwrote {len(pairs)} DPO pairs -> {out}")

    save_result("exp11_best_of_n", {
        "model": args.model, "usable": usable, "floor": floor,
        "n_summaries": args.n_summaries,
        "immediate_d": {"best": avg(best_d), "random": avg(rand_d)},
        "grounded_agree": {"best": avg(best_ag), "random": avg(rand_ag)},
        "downstream_d": {"best": avg(down_best), "random": avg(down_rand)},
        "raw": {"best_d": best_d, "rand_d": rand_d,
                "down_best": down_best, "down_rand": down_rand,
                "floors": floors},
    })


if __name__ == "__main__":
    main()
