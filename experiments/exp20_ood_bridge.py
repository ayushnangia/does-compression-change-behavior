"""EXP 20 (GPU) — The bridge: does input-side out-of-distribution-ness of a
compacted context PREDICT output-side behavioral damage?

Zhang & Khattab (2026) show harnesses generalize by keeping LM calls
in-distribution, measured with input-side surface metrics. We measure
output-side behavioral divergence. This experiment pairs them: for each
compressed context, compute
  input side:  3-gram containment vs the original context, and mean
               per-token NLL of the compacted context under the model
  output side: coarse divergence D, acting rate
across conditions spanning both content and format variation (including
summary vs summary_native: same content, different format). If input-side
OOD-ness correlates with D, "keep the compacted state in-distribution"
becomes a predictive design rule rather than two descriptive findings.

    python exp20_ood_bridge.py --examples-file ../examples_16k_large.json
"""

from __future__ import annotations

import argparse
import random

from common import load_model, save_result, REPO  # noqa: F401
from stats import mean
from behavior import sample_actions
from data import load_examples_file
from exp4_block_ablation import BLOCK_RES, random_trim
from compressors import SUMMARIZE, _generate, _ids
from metrics import (acting_rate, action_change_tools, ngram_containment)

WRAP_NOTE = ("<turn index=0 role=user>\n<content>\n[Summary of work so far: "
             "{s}]\n</content>\n</turn>\n")
WRAP_TRACE = ("<turn index=0 role=tool>\n<content>\n[Context compacted. "
              "Summary of prior work: {s}]\n</content>\n</turn>\n")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / max(vx * vy, 1e-9)


def mean_nll(model, ids, device):
    import torch
    x = torch.tensor([ids[-4096:]], device=device)   # cap for memory (8k OOMed alongside 16k sampling)
    with torch.no_grad():
        out = model(x, labels=x)
    return float(out.loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=20)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    ids = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]
    examples = load_examples_file(args.examples_file, args.num_examples)
    conds = ["keep_recent", "random", "block_aware", "summary_note",
             "summary_trace", "wrapper_only"]

    rows = {c: {"contain": [], "nll": [], "coarse": [], "acting": []}
            for c in conds}
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        rng = random.Random(base)
        old = ex.context_ids[: -len(ex.recent_ids)]
        old_text = tokenizer.decode(old, skip_special_tokens=True)
        budget = max(64, int(len(old) * args.rate))

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped")
            continue
        usable += 1

        summ = None
        for c in conds:
            if c == "keep_recent":
                comp = old[-budget:]
            elif c == "random":
                comp = random_trim(list(old), budget, rng)
            elif c == "block_aware":
                t = BLOCK_RES["drop_reasoning"].sub(lambda m: m.group(1), old_text)
                t = BLOCK_RES["drop_observations"].sub(lambda m: m.group(1), t)
                comp = random_trim(ids(t), budget, rng)
            elif c in ("summary_note", "summary_trace"):
                if summ is None:
                    gen = _generate(model, tokenizer,
                                    list(old) + _ids(tokenizer, SUMMARIZE),
                                    min(budget, 384), device)
                    summ = tokenizer.decode(gen, skip_special_tokens=True).strip()
                wrap = WRAP_NOTE if c == "summary_note" else WRAP_TRACE
                comp = ids(wrap.format(s=summ))
            else:  # wrapper_only
                comp = ids(WRAP_NOTE.format(s=""))

            comp_text = tokenizer.decode(comp, skip_special_tokens=True)
            ctx = list(comp) + list(ex.recent_ids)
            acts = sample_actions(model, tokenizer, ctx, device,
                                  samples=args.samples, seed=base + 3)
            rows[c]["contain"].append(ngram_containment(comp_text, old_text))
            rows[c]["nll"].append(mean_nll(model, ctx, device))
            rows[c]["coarse"].append(action_change_tools(full_a, acts))
            rows[c]["acting"].append(acting_rate(acts))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    print(f"\nusable: {usable}   rate: {args.rate:.0%}\n")
    print(f"{'condition':14s} {'contain':>8s} {'nll':>6s} {'coarse-D':>9s} {'acting':>7s}")
    flat_x_c, flat_x_n, flat_d, flat_a = [], [], [], []
    summary_rows = {}
    for c in conds:
        r = rows[c]
        summary_rows[c] = {k: mean(v) for k, v in r.items()}
        print(f"{c:14s} {mean(r['contain']):8.2f} {mean(r['nll']):6.2f} "
              f"{mean(r['coarse']):9.2f} {mean(r['acting']):7.2f}")
        flat_x_c += r["contain"]; flat_x_n += r["nll"]
        flat_d += r["coarse"]; flat_a += r["acting"]
    print(f"\nspearman(containment, coarse-D): {spearman(flat_x_c, flat_d):+.2f}"
          f"   (containment, acting): {spearman(flat_x_c, flat_a):+.2f}")
    print(f"spearman(NLL, coarse-D):         {spearman(flat_x_n, flat_d):+.2f}"
          f"   (NLL, acting):         {spearman(flat_x_n, flat_a):+.2f}")

    save_result("exp20_ood_bridge", {
        "model": args.model, "usable": usable, "rate": args.rate,
        "conditions": summary_rows,
        "spearman": {"contain_vs_D": spearman(flat_x_c, flat_d),
                     "contain_vs_acting": spearman(flat_x_c, flat_a),
                     "nll_vs_D": spearman(flat_x_n, flat_d),
                     "nll_vs_acting": spearman(flat_x_n, flat_a)},
        "raw": rows,
    })


if __name__ == "__main__":
    main()
