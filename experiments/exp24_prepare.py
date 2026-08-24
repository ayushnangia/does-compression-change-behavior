"""EXP 24B (CPU) — prepare task-split data for extractive compactor GRPO.

The policy never writes memory text. It outputs block indices; the renderer
copies those blocks exactly. This turns the main empirical finding (verbatim >
rewritten) into a hard architectural guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from exp24_data import extract_turns, make_selector_prompt, stable_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B",
                    help="tokenizer that created the context IDs")
    ap.add_argument("--out-dir", default="results/exp24_data")
    ap.add_argument("--budget-fraction", type=float, default=0.25)
    ap.add_argument("--max-prompt-chars", type=int, default=24000)
    args = ap.parse_args()
    if not 0 < args.budget_fraction <= 1:
        raise SystemExit("budget fraction must be in (0,1]")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    examples = json.load(open(args.examples_file))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    handles = {s: (out / f"{s}.jsonl").open("w")
               for s in ("train", "validation", "test")}
    counts = {s: 0 for s in handles}
    tasks = {s: set() for s in handles}
    try:
        for i, ex in enumerate(examples):
            if not ex.get("logged_action"):
                continue
            recent_n = len(ex["recent_ids"])
            old_ids = ex["context_ids"][:-recent_n] if recent_n else ex["context_ids"]
            old_text = tok.decode(old_ids, skip_special_tokens=False)
            recent_text = tok.decode(ex["recent_ids"], skip_special_tokens=False)
            header, units = extract_turns(old_text)
            if len(units) < 2:
                continue
            budget = max(1, int(len(old_text) * args.budget_fraction))
            split = stable_split(ex["repo"])
            row = {
                "id": i, "task": ex["repo"], "source_model": args.model,
                # Conversational form makes TRL apply Qwen3.5's native chat
                # template. A plain string is treated as raw LM continuation;
                # pilot 813724 then copied trace fragments instead of JSON.
                "prompt": [{"role": "user", "content": make_selector_prompt(
                    units, budget, args.max_prompt_chars)}],
                "header": header, "units_json": json.dumps(units),
                "recent_text": recent_text, "logged_action": ex.get("logged_action"),
                "budget_chars": budget,
            }
            handles[split].write(json.dumps(row) + "\n")
            counts[split] += 1; tasks[split].add(ex["repo"])
    finally:
        for f in handles.values(): f.close()
    manifest = {
        "source_model": args.model,
        "source_file": str(Path(args.examples_file).resolve()),
        "source_sha256": hashlib.sha256(Path(args.examples_file).read_bytes()).hexdigest(),
        "budget_fraction": args.budget_fraction,
        "splits": {s: {"examples": counts[s], "tasks": sorted(tasks[s])}
                   for s in counts},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({s: {"examples": counts[s], "tasks": len(tasks[s])}
                      for s in counts}, indent=2))


if __name__ == "__main__":
    main()
