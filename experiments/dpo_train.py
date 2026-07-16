"""E-B step 1 (GPU) — Train the compressor by preference optimization on
D-ranked summary pairs. No critic, no GRPO, no rollouts: the entire
CompactionRL optimization stack replaced by offline DPO on a behavioral
signal (see PLAN.md).

Input: summary_pairs.jsonl from exp11 (chosen = low-D, rejected = high-D).
Output: a LoRA adapter for the summarizer model.

    python dpo_train.py --pairs results/summary_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B",
                    help="small summarizer to train (executor stays 9B)")
    ap.add_argument("--pairs", default="results/summary_pairs.jsonl")
    ap.add_argument("--out", default="results/dpo_compressor")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--min-gap", type=float, default=0.25,
                    help="discard pairs whose D-gap is within sampling noise")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (debug matrix)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    rows = [json.loads(l) for l in Path(args.pairs).read_text().splitlines()]
    # keep only pairs whose D-gap exceeds measurement noise: with 8-sample
    # scoring the floor is ~0.35, so a small gap is noise, and DPO on
    # noise-ranked pairs learns noise.
    kept = [r for r in rows if r["d_rejected"] - r["d_chosen"] >= args.min_gap]
    print(f"{len(kept)}/{len(rows)} pairs pass the D-gap filter "
          f"(min_gap={args.min_gap}); mean gap of kept: "
          f"{sum(r['d_rejected'] - r['d_chosen'] for r in kept) / max(len(kept), 1):.2f}")
    if len(kept) < 8:
        raise SystemExit("too few reliable pairs — generate more (exp11 with "
                         "more examples/samples) before training")
    rows = kept
    # Full-length prompts: the compressor must train on the compaction
    # regime, not truncated snippets. The memory fix is architectural, not
    # truncation: Liger fused CE never materializes the (seq x 150k-vocab)
    # logits tensor, ref log-probs are precomputed in a separate pass, and
    # activations offload to CPU.
    ds = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"],
          "rejected": r["rejected"]} for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.config.use_cache = False

    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj", "gate_proj", "up_proj",
                                          "down_proj"])
    cuda = torch.cuda.is_available()   # falls back cleanly for CPU testing
    cfg = DPOConfig(
        output_dir=args.out, num_train_epochs=args.epochs, beta=args.beta,
        learning_rate=args.lr, per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # MP-safe
        bf16=cuda, logging_steps=5, save_strategy="no",
        max_length=4096, truncation_mode="keep_end", report_to=[],
        # Constraint matrix discovered en route (each by test or one run):
        #   liger-DPO x PEFT -> NotImplementedError
        #   liger-DPO x precompute_ref_log_probs -> ValueError
        #   precompute_ref_log_probs -> datasets-5.0 arrow cache bug
        #   activation_offloading x 2-GPU device_map -> cross-device backward
        # Resolution: the BORING path. 2 GPUs for headroom (weights split,
        # full-length logits fit on the lm_head GPU), plain PEFT-DPO with
        # non-reentrant checkpointing, no exotic memory flags.
        # default fused adamw: LoRA optimizer states are tiny; bitsandbytes
        # would add an import-risk dependency for ~no memory benefit
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tokenizer, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
