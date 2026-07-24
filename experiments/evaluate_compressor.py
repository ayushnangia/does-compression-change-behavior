"""E-B step 2 (GPU) — Held-out evaluation: does the DPO-trained compressor
beat its own base model as a summarizer?

Summarizer: Qwen3.5-4B (base vs base+DPO adapter).
Executor: Qwen3.5-9B, fixed — so any gain is in the SUMMARY, and because
summarizer != executor, a win here is already evidence of portability (E-C).

Held-out examples: indices >= --train-cutoff of the examples file (exp11
built pairs from the first --train-cutoff examples).

    python evaluate_compressor.py --examples-file ../data/examples_64.json \
        --adapter results/dpo_compressor
"""

from __future__ import annotations

import argparse

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
    ap.add_argument("--executor", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--summarizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--train-cutoff", type=int, default=24)
    ap.add_argument("--num-eval", type=int, default=24)
    ap.add_argument("--heldout-tasks", default=None,
                    help="JSON list of held-out task names -> TASK-level "
                         "split (overrides index cutoff)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--summary-samples", type=int, default=2,
                    help="summaries per variant per example (averages out "
                         "single-draw summary variance)")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    ex_tok, executor, device = load_model(args.executor)
    print(f"loading summarizer {args.summarizer} + adapter ...")
    base_sum = AutoModelForCausalLM.from_pretrained(
        args.summarizer, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()
    # wrap ONCE; toggle adapter layers per variant (cleaner than lazy wrap)
    summarizer = PeftModel.from_pretrained(base_sum, args.adapter).eval()

    all_ex = load_examples_file(args.examples_file, None)
    if args.heldout_tasks:
        import json as _json
        held = set(_json.load(open(args.heldout_tasks)))
        examples = [e for e in all_ex if e.repo in held][: args.num_eval]
        print(f"{len(examples)} examples from {len(held)} HELD-OUT tasks")
    else:
        examples = all_ex[args.train_cutoff:args.train_cutoff + args.num_eval]
        print(f"{len(examples)} held-out examples (index cutoff "
              f"{args.train_cutoff})")

    variants = ("base", "dpo")
    stats = {v: {"d": [], "acting": [], "agree": []} for v in variants}
    floors = []
    usable = 0

    def summarize(ids, seed, use_adapter):
        if use_adapter:
            summarizer.enable_adapter_layers()
        else:
            summarizer.disable_adapter_layers()
        prompt = list(ids) + _ids(ex_tok, SUMMARIZE)
        return sample_texts(summarizer, ex_tok, prompt, summarizer.device,
                            samples=1, max_new=args.budget, temperature=0.7,
                            seed=seed)[0].strip()

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + (ei + 1000) * 17
        old = ex.context_ids[: -len(ex.recent_ids)]

        full_a = sample_actions(executor, ex_tok, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped")
            continue
        usable += 1
        full_b = sample_actions(executor, ex_tok, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change(full_a, full_b))

        for v in variants:
            ds, acs, ags = [], [], []
            for si in range(args.summary_samples):
                text = summarize(old, base + 5 + si * 37,
                                 use_adapter=(v == "dpo"))
                ctx = _ids(ex_tok, WRAP.format(s=text)) + ex.recent_ids
                acts = sample_actions(executor, ex_tok, ctx, device,
                                      samples=args.samples, seed=base + 3)
                ds.append(action_change(full_a, acts))
                acs.append(acting_rate(acts))
                ag = agreement(acts, ex.logged_action)
                if ag is not None:
                    ags.append(ag)
            stats[v]["d"].append(sum(ds) / len(ds))
            stats[v]["acting"].append(sum(acs) / len(acs))
            if ags:
                stats[v]["agree"].append(sum(ags) / len(ags))
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\nusable: {usable}   floor: {floor:.3f}   "
          f"(summarizer {args.summarizer.split('/')[-1]}, "
          f"executor {args.executor.split('/')[-1]})\n")
    print(f"{'summarizer':10s} {'D (CI)':>16s} {'acting (CI)':>16s} {'agree':>6s}")
    rows = {}
    for v in variants:
        rows[v] = {"d": avg(stats[v]["d"]), "acting": avg(stats[v]["acting"]),
                   "agree": avg(stats[v]["agree"])}
        print(f"{v:10s} {fmt_ci(stats[v]['d']):>16s} "
              f"{fmt_ci(stats[v]['acting']):>16s} {rows[v]['agree']:6.2f}")
    if usable > 1:
        print(f"paired p (D, dpo vs base): "
              f"{paired_permutation_p(stats['dpo']['d'], stats['base']['d']):.3f}")

    save_result("expB_dpo_eval", {
        "executor": args.executor, "summarizer": args.summarizer,
        "usable": usable, "floor": floor, "variants": rows,
        "raw": {v: stats[v] for v in variants},
    })


if __name__ == "__main__":
    main()
