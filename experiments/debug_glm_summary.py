"""Verify or refute the quarantined GLM exp7 result: does GLM-4.7-Flash
genuinely stop acting after summary contexts (acting 0.00), or does it
switch to a tool-call syntax our parser misses?

Dumps raw continuations of summary-wrapped contexts + what the parser sees.

    python debug_glm_summary.py --examples-file ../examples_glm.json
"""

from __future__ import annotations

import argparse

from common import load_model, REPO  # noqa: F401
from behavior import parse_action, sample_texts
from compressors import SUMMARIZE, _generate, _ids
from data import load_examples_file

WRAP = ("<turn index=0 role=user>\n<content>\n[Summary of work so far: "
        "{s}]\n</content>\n</turn>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=3)
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)

    for ei, ex in enumerate(examples):
        old = ex.context_ids[: -len(ex.recent_ids)]
        gen = _generate(model, tokenizer, old + _ids(tokenizer, SUMMARIZE),
                        256, device)
        summ = tokenizer.decode(gen, skip_special_tokens=True).strip()
        ctx = _ids(tokenizer, WRAP.format(s=summ)) + ex.recent_ids
        print(f"\n{'='*70}\nEXAMPLE {ei} \u2014 summary head: {summ[:150]!r}")
        for si, t in enumerate(sample_texts(model, tokenizer, ctx, device,
                                            samples=args.samples, seed=ei)):
            print(f"--- cont {si} (parsed: {parse_action(t)!r}) ---")
            print(t[:900])


if __name__ == "__main__":
    main()
