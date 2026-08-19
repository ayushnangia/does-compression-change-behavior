"""Build many Qwen3.8-on-policy decision points for exp24 efficiently.

Unlike the older builder, each trajectory is tokenized once and character
anchors are mapped through the fast tokenizer's offset table. This avoids
re-tokenizing every prefix (quadratic work) when collecting 1,000+ points.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from behavior import parse_action
from data import Example, _serialize, save_examples
from prefetch_onpolicy import trajectory_to_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--traj-glob", required=True)
    ap.add_argument("--context-tokens", type=int, default=65536)
    ap.add_argument("--min-context-tokens", type=int, default=8192)
    ap.add_argument("--recent-tokens", type=int, default=4096)
    ap.add_argument("--num-examples", type=int, default=2000)
    ap.add_argument("--max-per-task", type=int, default=100)
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--out", default="data/examples_qwen38_exp24.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("exp24 efficient builder requires a fast tokenizer")
    paths = sorted(glob.glob(args.traj_glob))
    if not paths:
        raise SystemExit(f"no trajectories match {args.traj_glob}")
    out, per_task = [], {}
    for pi, path in enumerate(paths):
        if len(out) >= args.num_examples:
            break
        row = trajectory_to_row(path); task = row["repo"]
        text = _serialize(row)
        anchors = [m.start() for m in re.finditer(r"<tool_calls>", text)]
        turn_chars = [m.start() for m in re.finditer(r"<turn\b", text)]
        if not anchors or not turn_chars:
            continue
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        all_ids = enc["input_ids"]; offsets = enc["offset_mapping"]
        ends = [end for _, end in offsets]
        anchor_tok = {a: bisect.bisect_left(ends, a + 1) for a in anchors}
        turn_toks = sorted({bisect.bisect_left(ends, c + 1) for c in turn_chars})
        rng = random.Random(args.seed * 100003 + pi); rng.shuffle(anchors)
        for a in anchors:
            if len(out) >= args.num_examples or per_task.get(task, 0) >= args.max_per_task:
                break
            ti = anchor_tok[a]
            if ti < args.min_context_tokens:
                continue
            # Both cuts begin at complete <turn> boundaries. This prevents
            # malformed half-turn prefixes from becoming an unbudgeted header
            # or a malformed "recent" segment.
            context_target = max(0, ti - args.context_tokens)
            ci = bisect.bisect_left(turn_toks, context_target)
            if ci < len(turn_toks) and turn_toks[ci] < ti:
                start = turn_toks[ci]
            else:
                start = turn_toks[max(0, bisect.bisect_left(turn_toks, ti) - 1)]
            recent_target = max(start, ti - args.recent_tokens)
            ri = bisect.bisect_left(turn_toks, recent_target)
            if ri < len(turn_toks) and turn_toks[ri] < ti:
                recent_start = turn_toks[ri]
            else:
                recent_start = turn_toks[max(0, bisect.bisect_left(turn_toks, ti) - 1)]
            window = all_ids[start:ti]
            if len(window) > args.context_tokens + 2048 or ti - recent_start > args.recent_tokens + 2048:
                continue  # pathological giant turn; do not silently break budget
            logged = parse_action(text[a:a + 12000])
            if logged is None:
                continue
            later = sorted(x for x in anchors if x > a)[:3]
            futures = []
            for nxt in later:
                seg = all_ids[ti:anchor_tok[nxt]]
                if len(seg) > 1536:
                    break
                futures.append(seg)
                ti = anchor_tok[nxt]
            out.append(Example(context_ids=window,
                recent_ids=all_ids[recent_start:anchor_tok[a]], repo=task,
                logged_action=logged, future_segments=futures))
            per_task[task] = per_task.get(task, 0) + 1
    print(f"{len(out)} examples from {len(per_task)} Qwen3.8 tasks")
    if len(out) < 100:
        print("WARNING: fewer than 100 points; plumbing only, not training evidence")
    save_examples(out, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
