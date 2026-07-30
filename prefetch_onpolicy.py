"""Build ON-POLICY decision-point examples from our own Terminal-Bench
trajectories (the model being measured generated these traces itself \u2014
closing the off-policy confound).

Run on the login node (tokenizer only, no GPU, no internet needed):

    python prefetch_onpolicy.py --out data/examples_onpolicy.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re

from data import Example, save_examples, _serialize
from behavior import parse_action


def trajectory_to_row(path: str) -> dict:
    """Convert a Terminus-2 trajectory.json into the row shape _serialize
    expects (role/content/tool_calls/observation messages)."""
    d = json.load(open(path))
    msgs = []
    for s in d.get("steps", []):
        msgs.append({
            "role": s.get("source", "event"),
            "content": s.get("message"),
            "tool_calls": s.get("tool_calls"),
            "observation": s.get("observation"),
        })
    task = path.split("/")[-3].split("__")[0]
    return {"instance_id": task, "repo": task, "trajectory": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--traj-glob",
                    default="/scratch/anangia/tb2/jobs/tb2-qwen35-9b/*/agent/trajectory.json")
    ap.add_argument("--context-tokens", type=int, default=4096)
    ap.add_argument("--recent-tokens", type=int, default=512)
    ap.add_argument("--num-examples", type=int, default=64)
    ap.add_argument("--max-per-task", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/examples_onpolicy.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    paths = sorted(glob.glob(args.traj_glob))
    print(f"{len(paths)} on-policy trajectories")
    out, per_task = [], {}

    for pi, path in enumerate(paths):
        if len(out) >= args.num_examples:
            break
        row = trajectory_to_row(path)
        repo = row["repo"]
        if per_task.get(repo, 0) >= args.max_per_task:
            continue
        text = _serialize(row)
        anchors = [m.start() for m in re.finditer(r"<tool_calls>", text)]
        if not anchors:
            continue
        rng = random.Random(args.seed * 100003 + pi)
        rng.shuffle(anchors)
        for a in anchors:
            if len(out) >= args.num_examples or per_task.get(repo, 0) >= args.max_per_task:
                break
            ids = tokenizer(text[:a], add_special_tokens=False)["input_ids"]
            if len(ids) < args.context_tokens:
                continue
            window = ids[-args.context_tokens:]
            logged = parse_action(text[a:a + 2000])
            futures = []
            later = sorted(x for x in anchors if x > a)[:3]
            bounds = [a] + later
            for i in range(len(later)):
                seg = tokenizer(text[bounds[i]:bounds[i + 1]],
                                add_special_tokens=False)["input_ids"]
                if len(seg) > 1536:
                    break
                futures.append(seg)
            out.append(Example(context_ids=window,
                               recent_ids=window[-args.recent_tokens:],
                               repo=repo, logged_action=logged,
                               future_segments=futures))
            per_task[repo] = per_task.get(repo, 0) + 1

    print(f"{len(out)} examples from {len(per_task)} tasks; "
          f"{sum(1 for e in out if e.logged_action)} with logged actions")
    save_examples(out, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
