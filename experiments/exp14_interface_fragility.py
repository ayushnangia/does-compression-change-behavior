"""EXP 14 (GPU) — Interface-fragility matrix: is GLM's collapse on
compaction-note contexts a mode conflict, a sampling artifact, or model
fragility? And is a similar-scale Qwen robust under IDENTICAL conditions?

Design: {model} x {our trace-style wrapper, the model's NATIVE chat
template} x {top_p 1.0, 0.9}. For each cell: acting rate (parse success)
over sampled continuations + raw dumps for qualitative reading.

Readings:
  native-template coherent, wrapper salad -> MODE CONFLICT: compaction notes
      must be delivered in the model's own format (crisp design rule).
  salad everywhere for GLM only          -> model fragility on summarized
      context (deeper finding).
  coherent at top_p=0.9 only             -> our full-distribution sampling
      amplified a marginal effect (honesty caveat for exp5/exp7 numbers).

    python exp14_interface_fragility.py --model zai-org/GLM-4.7-Flash \
        --examples-file ../data/examples_glm.json
"""

from __future__ import annotations

import argparse

from common import load_model, save_result, REPO  # noqa: F401
from behavior import parse_action, sample_texts
from compressors import SUMMARIZE, _generate, _ids
from data import load_examples_file

WRAP = ("<turn index=0 role=user>\n<content>\n[Summary of work so far: "
        "{s}]\n</content>\n</turn>\n")


def build_contexts(ex, summ, tokenizer):
    """(name, ids) for both delivery formats of the SAME summary."""
    recent_text = tokenizer.decode(ex.recent_ids, skip_special_tokens=True)
    wrapper_ids = _ids(tokenizer, WRAP.format(s=summ)) + list(ex.recent_ids)
    msgs = [{"role": "user", "content":
             f"Earlier work was compacted. Summary of work so far: {summ}\n\n"
             f"Most recent trace (continue from here, reply with the next "
             f"tool call in the same format as the trace):\n{recent_text}"}]
    out = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=True)
    # returns list[int], BatchEncoding, or tokenizers.Encoding depending on
    # tokenizer version — coerce to a plain list of ints
    if hasattr(out, "ids"):
        native_ids = list(out.ids)
    elif isinstance(out, dict) or hasattr(out, "input_ids"):
        native_ids = list(out["input_ids"])
    else:
        native_ids = list(out)
    if native_ids and isinstance(native_ids[0], list):   # batched form
        native_ids = native_ids[0]
    return [("wrapper", wrapper_ids), ("native", native_ids)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=6)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--top-ps", type=float, nargs="+", default=[1.0, 0.9])
    ap.add_argument("--dump-chars", type=int, default=400)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)

    cells = {}
    for ei, ex in enumerate(examples):
        old = ex.context_ids[: -len(ex.recent_ids)]
        gen = _generate(model, tokenizer, old + _ids(tokenizer, SUMMARIZE),
                        256, device)
        summ = tokenizer.decode(gen, skip_special_tokens=True).strip()
        for fmt, ctx in build_contexts(ex, summ, tokenizer):
            for tp in args.top_ps:
                texts = sample_texts(model, tokenizer, ctx, device,
                                     samples=args.samples, top_p=tp, seed=ei)
                acts = [parse_action(t) for t in texts]
                key = f"{fmt}@top_p={tp:g}"
                cells.setdefault(key, []).extend(a is not None for a in acts)
                if ei < 2:   # qualitative dumps for the first two examples
                    print(f"\n--- ex{ei} {key} (parsed: {acts[0]!r}) ---")
                    print(texts[0][: args.dump_chars])
        print(f"example {ei}: done")

    print(f"\n=== acting rate (parse success) \u2014 {args.model.split('/')[-1]} ===")
    print(f"{'cell':22s} {'acting':>7s} {'n':>4s}")
    rows = {}
    for k, v in sorted(cells.items()):
        rows[k] = sum(v) / len(v)
        print(f"{k:22s} {rows[k]:7.2f} {len(v):4d}")

    save_result(f"exp14_fragility_{args.model.split('/')[-1].replace('.', '')}",
                {"model": args.model, "cells": rows,
                 "n_examples": len(examples), "samples": args.samples})


if __name__ == "__main__":
    main()
