"""EXP 23 (GPU) — One-liner action history + verbatim recent tail, at
matched budgets R in {25%, 50%, 75%}.

The question exp21 left open: the wrapped canonical action history is nearly
free (~870 tokens for a 16k history) and keep_recent is the incumbent
champion at normal budgets. Does PREPENDING the full one-liner skeleton to a
verbatim tail beat the tail alone, at the SAME total budget? This is the
offline twin of exp22's arm B (OneLinerTerminus: canonical history +
verbatim tail), so its verdict transfers to the live-episode arm.

Conditions, all at old-history budget B = R x |old| (budget accounting is
exact; the exp22 budget-confound lesson - policies decide WHAT fills the
budget, never HOW MUCH):
  keep_recent        old[-B:]                                   (incumbent)
  oneliner_tail      wrapped canonical lines of the FULL history, capped at
                     B/2 (drop OLDEST lines first), + verbatim tail filling
                     the remainder                               (the hybrid)
  skeleton_tail      same geometry but raw <tool_calls> blocks instead of
                     canonical lines (separates "add action history" from
                     "canonicalize it"; containment-law control)
  canonical_wrapped  one-liners only at budget B                 (exp21 ref)

Disclosed by design: the tail's raw text repeats the newest calls that also
appear in the skeleton (identical to exp22 arm B delivery).

Pre-registered predictions (stated before running):
  P1 (PRIMARY): agreement(oneliner_tail) > agreement(keep_recent) at R=0.25,
     paired permutation across examples. Rationale: the tail lacks the
     distant action skeleton; adding it costs ~2% of budget.
  P2: the hybrid-minus-tail gap SHRINKS monotonically in R (at 75% the tail
     already contains most of the action history).
  P3 (exploratory): skeleton_tail >= oneliner_tail at these moderate rates
     (exp20/exp20b containment law: canonicalizing destroys verbatim
     n-grams; exp21 found raw beats canonical by ~5-9pts at moderate R).

Reporting standard (AUDIT metric-critique adoption note): action_change_all
(3 granularities) + debiased excess/p + harm_score + acting + agreement,
all floor-referenced; raw per-example arrays saved.

    python exp23_oneliner_tail.py --examples-file ../data/examples_onpolicy.json
"""

from __future__ import annotations

import argparse
import random

from common import load_model, save_result, REPO  # noqa: F401
from stats import fmt_ci, bootstrap_ci, paired_permutation_p  # noqa: F401
from behavior import sample_actions
from data import load_examples_file
from metrics import (acting_rate, action_change_tools, action_change_all,
                     debiased_change, harm_score)
from exp21_canonical_skeleton import (TOOLCALL_BLOCK, _parse_calls,
                                      canonicalize, agreement,
                                      build as exp21_build)


def build_units(old_text: str, tokenizer, canonical: bool):
    """Per-call token-id units, oldest first. canonical=True -> one-liner
    shorthand; False -> raw <tool_calls> blocks. Native wrapper always kept
    (exp21 P3: the wrapper is load-bearing)."""
    blocks = TOOLCALL_BLOCK.findall(old_text)
    if canonical:
        texts = [f"<tool_calls>{canonicalize(name, args)}</tool_calls>\n"
                 for b in blocks for name, args in _parse_calls(b)]
    else:
        texts = [f"<tool_calls>{b}</tool_calls>\n" for b in blocks]
    return [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


def fit_hybrid(old_ids, unit_ids, budget: int):
    """Skeleton capped at budget//2 keeping the NEWEST units (oldest dropped
    first); verbatim tail of old_ids fills every remaining token. Exact
    accounting: len(result) <= budget always.
      returns (ids, skeleton_tokens, tail_tokens)"""
    cap = budget // 2
    kept, used = [], 0
    for u in reversed(unit_ids):          # newest first
        if used + len(u) > cap:
            break
        kept.append(u)
        used += len(u)
    kept.reverse()                        # restore temporal order
    skel = [t for u in kept for t in u]
    tail_budget = budget - len(skel)
    tail = list(old_ids[-tail_budget:]) if tail_budget > 0 else []
    return skel + tail, len(skel), len(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=24)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)
    conds = ["keep_recent", "oneliner_tail", "skeleton_tail", "canonical_wrapped"]

    stats = {(c, r): {"acting": [], "coarse": [], "all": [], "excess": [],
                      "harm": [], "agree": [], "skel": [], "tail": [], "used": []}
             for c in conds for r in args.rates}
    floors, usable = [], 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        rng = random.Random(base)
        old = ex.context_ids[: -len(ex.recent_ids)]
        old_text = tokenizer.decode(old, skip_special_tokens=True)

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped (acting<0.5)")
            continue
        usable += 1
        full_b = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change_tools(full_a, full_b))

        units_canon = build_units(old_text, tokenizer, canonical=True)
        units_raw = build_units(old_text, tokenizer, canonical=False)

        for r in args.rates:
            budget = max(16, int(len(old) * r))
            for c in conds:
                skel_n = tail_n = 0
                if c == "keep_recent":
                    comp = list(old[-budget:])
                    tail_n = len(comp)
                elif c == "oneliner_tail":
                    comp, skel_n, tail_n = fit_hybrid(old, units_canon, budget)
                elif c == "skeleton_tail":
                    comp, skel_n, tail_n = fit_hybrid(old, units_raw, budget)
                else:  # canonical_wrapped (exp21 continuity)
                    comp = exp21_build("canonical_wrapped", old_text, budget,
                                       tokenizer, rng)
                assert len(comp) <= budget, f"budget violated: {c} R={r}"
                acts = sample_actions(model, tokenizer,
                                      list(comp) + list(ex.recent_ids), device,
                                      samples=args.samples, seed=base + 3)
                cell = stats[(c, r)]
                cell["acting"].append(acting_rate(acts))
                cell["coarse"].append(action_change_tools(full_a, acts))
                cell["all"].append(action_change_all(full_a, acts))
                exc, p = debiased_change(full_a, acts, seed=base)
                cell["excess"].append(exc)
                cell["harm"].append(harm_score(full_a, acts, ex.logged_action))
                cell["skel"].append(skel_n)
                cell["tail"].append(tail_n)
                cell["used"].append(len(comp))
                ag = agreement(acts, ex.logged_action)
                if ag is not None:
                    cell["agree"].append(ag)
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable: {usable}   coarse floor: {floor:.3f}")
    print("cells: coarse-D | acting | agreement | skel/tail tokens\n")
    rows = {}
    for c in conds:
        cells = []
        for r in args.rates:
            s = stats[(c, r)]
            n = len(s["acting"])
            co, ac = sum(s["coarse"]) / n, sum(s["acting"]) / n
            ag = sum(s["agree"]) / len(s["agree"]) if s["agree"] else float("nan")
            sk, tl = sum(s["skel"]) / n, sum(s["tail"]) / n
            cells.append(f" R={r:g}: {co:.2f}|{ac:.2f}|{ag:.2f}|{sk:.0f}/{tl:.0f}t")
            rows[f"{c}@{r:g}"] = {
                "coarse": co, "acting": ac, "agree": ag,
                "agree_ci": bootstrap_ci(s["agree"]) if s["agree"] else None,
                "skel_tokens": sk, "tail_tokens": tl,
                "mean_excess": sum(s["excess"]) / n,
                "mean_halt_increase": sum(h["halt_increase"] for h in s["harm"]) / n,
            }
        print(f"{c:18s}" + "".join(cells))

    # P1 (primary, pre-registered): hybrid vs incumbent at the lowest rate
    r0 = args.rates[0]
    p1 = paired_permutation_p(stats[("oneliner_tail", r0)]["agree"],
                              stats[("keep_recent", r0)]["agree"])
    print(f"\nP1 paired p (agreement @R={r0:g}, oneliner_tail vs keep_recent): {p1:.4f}")
    # P2: gap monotonicity across rates
    gaps = []
    for r in args.rates:
        ga = stats[("oneliner_tail", r)]["agree"]
        gb = stats[("keep_recent", r)]["agree"]
        m = min(len(ga), len(gb))
        gaps.append(sum(ga[:m]) / m - sum(gb[:m]) / m if m else float("nan"))
    print("P2 hybrid-minus-tail agreement gap by rate: "
          + "  ".join(f"R={r:g}:{g:+.3f}" for r, g in zip(args.rates, gaps)))
    # P3 (exploratory): raw vs canonical skeleton in the hybrid
    p3 = paired_permutation_p(stats[("skeleton_tail", r0)]["agree"],
                              stats[("oneliner_tail", r0)]["agree"])
    print(f"P3 paired p (agreement @R={r0:g}, skeleton_tail vs oneliner_tail): {p3:.4f}")

    save_result("exp23_oneliner_tail", {
        "model": args.model, "usable": usable, "floor": floor,
        "rates": args.rates, "cells": rows,
        "p1_primary": p1, "p2_gaps": gaps, "p3": p3,
        "raw": {f"{c}@{r:g}": stats[(c, r)] for c in conds for r in args.rates},
    })


if __name__ == "__main__":
    main()
