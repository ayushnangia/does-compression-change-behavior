"""Diagnostic: WHY does the model never 'act' on our decision points?

exp3-5 skipped all examples ("full-context model rarely acts") with
Qwen3.5-9B. Before changing thresholds or parsers, look at the raw
continuations: does the model produce tool calls in a different syntax
(parser problem), produce prose/reasoning first (max_new problem), or
genuinely not act (model/format mismatch)?

    python debug_continuations.py --examples-file ../examples_prefetched.json
"""

from __future__ import annotations

import argparse

from common import load_model, REPO  # noqa: F401
from behavior import parse_action, sample_texts
from data import load_examples_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=3)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=768)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)

    for ei, ex in enumerate(examples):
        tail = tokenizer.decode(ex.context_ids[-300:], skip_special_tokens=True)
        print(f"\n{'='*70}\nEXAMPLE {ei}  (context ends with:)\n...{tail[-400:]}")
        texts = sample_texts(model, tokenizer, ex.context_ids, device,
                             samples=args.samples, max_new=args.max_new,
                             seed=ei)
        for si, t in enumerate(texts):
            print(f"\n--- continuation {si} (parsed action: {parse_action(t)!r}) ---")
            print(t[:1200])


if __name__ == "__main__":
    main()
