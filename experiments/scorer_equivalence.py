"""T1 acceptance test: is the vLLM scorer behaviorally equivalent to the HF
reference, and how much faster is it?

Equivalence is DISTRIBUTIONAL (different RNGs -> different tokens): for each
example, D(hf_ref, backend) must behave like a same-context resample, i.e.
|D_hf_vs_hf2 - D_hf_vs_vllm| within the sampling floor on average.

Run once per backend (different venvs), then compare:
    ENV-compress:  python scorer_equivalence.py --backend hf   --out /tmp/eq_hf.json
    ENV-vllm:      python scorer_equivalence.py --backend vllm --out /tmp/eq_vllm.json
    (either):      python scorer_equivalence.py --compare /tmp/eq_hf.json /tmp/eq_vllm.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from data import load_examples_file  # noqa: E402


def run_backend(args):
    examples = load_examples_file(args.examples_file, args.num_examples)
    results = []
    if args.backend == "hf":
        from common import load_model
        from behavior import sample_actions
        tok, model, device = load_model(args.model)
        t0 = time.time()          # MARGINAL throughput: exclude model load —
                                  # one-time init amortizes to ~0 at scale
        for ei, ex in enumerate(examples):
            base = ei * 17
            a = sample_actions(model, tok, ex.context_ids, device,
                               samples=args.samples, seed=base + 1)
            b = sample_actions(model, tok, ex.context_ids, device,
                               samples=args.samples, seed=base + 2)
            results.append({"a": a, "b": b})
            print(f"example {ei}: done")
    else:
        from vllm_scorer import VLLMScorer
        sc = VLLMScorer(args.model, max_model_len=args.max_model_len)
        t0 = time.time()          # exclude engine startup (see above)
        ctxs = [ex.context_ids for ex in examples]
        # one batched pass for each seed-set
        acts_a = sc.sample_actions_batch(ctxs, samples=args.samples, seed=1)
        acts_b = sc.sample_actions_batch(ctxs, samples=args.samples, seed=10_000)
        results = [{"a": a, "b": b} for a, b in zip(acts_a, acts_b)]
    elapsed = time.time() - t0
    payload = {"backend": args.backend, "elapsed_s": elapsed,
               "n": len(examples), "samples": args.samples, "results": results}
    Path(args.out).write_text(json.dumps(payload))
    print(f"{args.backend}: {elapsed:.0f}s for {len(examples)} examples "
          f"-> {args.out}")


def compare(f_hf, f_vllm):
    from metrics import action_change
    from stats import mean, bootstrap_ci
    hf = json.loads(Path(f_hf).read_text())
    vl = json.loads(Path(f_vllm).read_text())
    floors, cross = [], []
    for rh, rv in zip(hf["results"], vl["results"]):
        floors.append(action_change(rh["a"], rh["b"]))       # hf self-resample
        cross.append(action_change(rh["a"], rv["a"]))        # hf vs vllm
    floor, x = mean(floors), mean(cross)
    lo, hi = bootstrap_ci([c - f for c, f in zip(cross, floors)])
    speedup = hf["elapsed_s"] / max(vl["elapsed_s"], 1e-9)
    print(f"floor (hf self):      {floor:.3f}")
    print(f"cross (hf vs vllm):   {x:.3f}")
    print(f"excess (cross-floor): {x - floor:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
    print(f"throughput: hf {hf['elapsed_s']:.0f}s vs vllm {vl['elapsed_s']:.0f}s "
          f"-> {speedup:.1f}x")
    ok = hi <= 0.15 and speedup >= 5.0
    print("EQUIVALENCE + THROUGHPUT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm"])
    ap.add_argument("--compare", nargs=2, metavar=("HF_JSON", "VLLM_JSON"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", default="../examples_64.json")
    ap.add_argument("--num-examples", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", default="/tmp/eq.json")
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    run_backend(args)


if __name__ == "__main__":
    main()
