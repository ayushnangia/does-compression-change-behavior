"""Prefetch everything the experiment needs, on a node WITH internet.

Alliance/Compute Canada compute nodes have no internet access, so streaming
the dataset or downloading the model inside the job fails. Run this once on
the login node instead:

    python prefetch.py --model Qwen/Qwen3.5-9B \
        --num-examples 32 --context-tokens 4096 --recent-tokens 512

It (1) snapshots the model into the HF cache ($HF_HOME) and (2) builds the
decision-point examples and saves them to a JSON the job loads with
`run_experiment.py --examples-file`.
"""

from __future__ import annotations

import argparse

from data import load_examples, save_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--num-examples", type=int, default=32)
    ap.add_argument("--context-tokens", type=int, default=4096)
    ap.add_argument("--recent-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/examples_prefetched.json")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    print(f"downloading model snapshot: {args.model} ...")
    snapshot_download(args.model)
    print("model cached.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("building decision-point examples from nvidia/Open-SWE-Traces ...")
    examples = load_examples(
        tokenizer, context_tokens=args.context_tokens,
        recent_tokens=args.recent_tokens, num_examples=args.num_examples,
        seed=args.seed,
    )
    save_examples(examples, args.out)
    print(f"saved {len(examples)} examples -> {args.out}")


if __name__ == "__main__":
    main()
