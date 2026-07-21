"""EXP 19 (GPU, vLLM) - Validate the exact logprob-based next-tool
distribution against the sampled estimate, and measure how much floor noise
it removes.

For N examples: compute (a) sampled tool-level TV between full and compressed
(8 samples each, the current method) and (b) exact TV from logprob-scored
tool distributions. Report correlation, and the exact-method noise on
identical contexts (should be ~0 by construction).

    (ENV-vllm2)  python exp19_exact_distribution.py --examples-file ../examples_16k_large.json
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
from data import load_examples_file
from metrics import action_change_tools
from stats import mean
from vllm_scorer import VLLMScorer

TOOLS = ["bash_command", "write_file", "str_replace", "task_complete"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=20480)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sc = VLLMScorer(args.model, max_model_len=args.max_model_len)
    exs = load_examples_file(args.examples_file, args.num_examples)

    sampled_tv, exact_tv = [], []
    for ei, ex in enumerate(exs):
        old = ex.context_ids[: -len(ex.recent_ids)]
        comp = old[len(old) // 2 :] + list(ex.recent_ids)   # keep_recent 50%
        full = list(ex.context_ids)
        # sampled (current method, tool level)
        a = sc.sample_actions(full, samples=8, seed=ei)
        b = sc.sample_actions(comp, samples=8, seed=ei + 999)
        sampled_tv.append(action_change_tools(a, b))
        # exact (new method)
        pf = sc.exact_tool_distribution(full, TOOLS, tok)
        pc = sc.exact_tool_distribution(comp, TOOLS, tok)
        exact_tv.append(0.5 * sum(abs(pf[t] - pc[t]) for t in TOOLS))
        print(f"example {ei}: sampled {sampled_tv[-1]:.2f} exact {exact_tv[-1]:.3f}")

    n = len(sampled_tv)
    mx, my = mean(sampled_tv), mean(exact_tv)
    cov = sum((x - mx) * (y - my) for x, y in zip(sampled_tv, exact_tv)) / n
    vx = sum((x - mx) ** 2 for x in sampled_tv) / n
    vy = sum((y - my) ** 2 for y in exact_tv) / n
    r = cov / max((vx * vy) ** 0.5, 1e-9)
    print(f"\nsampled mean {mx:.3f}  exact mean {my:.3f}  pearson r {r:.2f}")
    print("exact method has NO sampling floor: identical contexts give TV=0 by construction")

if __name__ == "__main__":
    main()
