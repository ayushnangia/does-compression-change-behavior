"""EXP 16 / T4 (GPU, vLLM) — Scaled on-policy DPO pair generation, with every
round-1 root cause fixed:

  RC1  candidates are written by the TRAINEE (4B), not the executor
  RC2  enough pairs for real optimization (target 200+ after gap filter)
  RC3  task-level train/held-out split recorded per pair
  +    marginalized reference: behavior averaged over two context lengths
       (exp3-v2: the target has positional sensitivity; average it out)

Phased for single-GPU batching (T1's 6.2x scorer):
  A. 4B engine: N candidate summaries per example, one batched pass
  B. 9B engine: reference behavior at 2 lengths + all candidate scorings,
     batched; fresh re-score of each argmin (winner's-curse guard)

    (ENV-vllm)  python exp16_pairs_at_scale.py --examples-file ../examples_16k_large.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path


def stable_hash(s: str) -> int:
    """Deterministic across processes (python's hash() is seed-randomized)."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from data import load_examples_file            # noqa: E402
from metrics import acting_rate, action_change # noqa: E402
from stats import mean, fmt_ci                 # noqa: E402
from vllm_scorer import VLLMScorer             # noqa: E402

SUMMARIZE = ("\n\n[Summarize the agent trace above so the work can be "
             "continued. Keep file paths, commands, errors, and decisions. "
             "Be concise.]\nSummary:")
WRAP = ("<turn index=0 role=user>\n<content>\n[Summary of work so far: "
        "{s}]\n</content>\n</turn>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--executor", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--trainee", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=280)
    ap.add_argument("--n-summaries", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--budget", type=int, default=384)
    ap.add_argument("--ref-lengths", type=int, nargs="+",
                    default=[16384, 8192])
    ap.add_argument("--heldout-mod", type=int, default=4,
                    help="tasks with hash%%mod==0 are HELD OUT of pairs")
    ap.add_argument("--max-model-len", type=int, default=20480)
    ap.add_argument("--pairs-out", default="results/summary_pairs_16k.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.executor, trust_remote_code=True)
    ids = lambda t: tok(t, add_special_tokens=False)["input_ids"]

    all_ex = load_examples_file(args.examples_file, None)
    train_ex = [e for e in all_ex
                if stable_hash(e.repo) % args.heldout_mod != 0][: args.num_examples]
    held = sorted({e.repo for e in all_ex
                   if stable_hash(e.repo) % args.heldout_mod == 0})
    print(f"{len(train_ex)} train-task examples; {len(held)} held-out tasks: "
          f"{held[:6]}...")

    # ---------- Phase A: trainee writes candidates (batched) ----------
    print(f"[A] {args.trainee} writing {args.n_summaries} candidates x "
          f"{len(train_ex)} examples ...")
    gen = VLLMScorer(args.trainee, max_model_len=args.max_model_len,
                     gpu_memory_utilization=0.85)
    prompts = [list(e.context_ids[: -len(e.recent_ids)]) + ids(SUMMARIZE)
               for e in train_ex]
    cand_texts = gen.sample_texts_batch(
        prompts, samples=args.n_summaries, max_new=args.budget,
        temperature=0.9, seed=5)
    del gen; gc.collect()
    import torch; torch.cuda.empty_cache()

    # ---------- Phase B: executor scores everything (batched) ----------
    print(f"[B] {args.executor} scoring ...")
    sc = VLLMScorer(args.executor, max_model_len=args.max_model_len,
                    gpu_memory_utilization=0.90)

    # references at two lengths (marginalized target) + a floor resample
    refs, floors_ctx = [], []
    for e in train_ex:
        refs.append([list(e.context_ids[-L:]) for L in args.ref_lengths])
    flat_refs = [c for r in refs for c in r]
    ref_acts_a = sc.sample_actions_batch(flat_refs, samples=args.samples, seed=1)
    ref_acts_b = sc.sample_actions_batch(flat_refs, samples=args.samples, seed=2)

    # candidate contexts
    cand_ctx, owner = [], []
    for i, (e, texts) in enumerate(zip(train_ex, cand_texts)):
        for t in texts:
            cand_ctx.append(ids(WRAP.format(s=t.strip())) + list(e.recent_ids))
            owner.append(i)
    cand_acts = sc.sample_actions_batch(cand_ctx, samples=args.samples, seed=3)

    # ---------- assemble pairs ----------
    nL = len(args.ref_lengths)
    pairs, floors, best_d, rand_d = [], [], [], []
    rescore_ctx, rescore_meta = [], []
    for i, e in enumerate(train_ex):
        ra = ref_acts_a[i * nL:(i + 1) * nL]
        rb = ref_acts_b[i * nL:(i + 1) * nL]
        if acting_rate(ra[0]) < 0.5:
            continue
        floors.append(mean([action_change(a, b) for a, b in zip(ra, rb)]))
        my = [j for j, o in enumerate(owner) if o == i]
        scored = []
        for j in my:
            d = mean([action_change(r, cand_acts[j]) for r in ra])  # marginalized
            scored.append((d, j))
        scored.sort()
        rand_d.append(mean([d for d, _ in scored]))
        (dbest, jbest), (dworst, jworst) = scored[0], scored[-1]
        rescore_ctx.append(cand_ctx[jbest])          # winner's-curse guard
        rescore_meta.append((i, ra))
        pairs.append({
            "task": e.repo,
            "prompt": tok.decode(e.context_ids[: -len(e.recent_ids)],
                                 skip_special_tokens=True)[-24000:] + "\n\n"
                      + SUMMARIZE.strip(),
            "chosen": cand_texts[i][jbest - my[0]].strip(),
            "rejected": cand_texts[i][jworst - my[0]].strip(),
            "d_chosen": dbest, "d_rejected": dworst,
        })
    fresh = sc.sample_actions_batch(rescore_ctx, samples=args.samples, seed=21)
    for f, (i, ra) in zip(fresh, rescore_meta):
        best_d.append(mean([action_change(r, f) for r in ra]))

    print(f"\nusable examples: {len(pairs)}   floor: {mean(floors):.3f}")
    print(f"select-by-D (fresh): {fmt_ci(best_d)}   random: {fmt_ci(rand_d)}")
    kept = [p for p in pairs if p["d_rejected"] - p["d_chosen"] >= 0.25]
    print(f"pairs passing 0.25 gap filter: {len(kept)}/{len(pairs)}")

    out = Path(args.pairs_out)
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for p in pairs:                        # write ALL; dpo_train filters
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(pairs)} pairs -> {out}")
    Path("results/heldout_tasks.json").write_text(json.dumps(held))


if __name__ == "__main__":
    main()
