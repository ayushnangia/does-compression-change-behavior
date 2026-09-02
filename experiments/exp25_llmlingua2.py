"""EXP 25 — frozen open-source LLMLingua-2 behavioral baseline.

This is exploratory and on-policy: source histories/logged actions come only
from Qwen3.8 Harbor trajectories. LLMLingua-2 compresses the old history; its
actual Qwen token count determines a matched-budget verbatim-recency control.
A frozen Qwen3.8 executor scores grounded next-action behavior through native
chat. No model is trained here.

Two stages allow one H100 to be reused sequentially:
  prepare: load LLMLingua-2, write compressed/matched contexts, exit
  score:   query an already-running Qwen3.8 vLLM server and aggregate
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from behavior import parse_action
from compressors import PATH_RE
from metrics import _verb

LINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
SOURCE_MODEL = "Qwen/Qwen3.8-27B"
CONDITIONS = ("full", "keep_recent_matched", "llmlingua2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(paths: list[Path], per_task: int) -> list[dict]:
    by_task: dict[str, list[dict]] = {}
    for path in paths:
        for line in path.open():
            row = json.loads(line)
            if row.get("source_model") != SOURCE_MODEL:
                raise SystemExit(f"OFF-POLICY ROW REFUSED in {path}")
            by_task.setdefault(row["task"], []).append(row)
    selected = []
    for task in sorted(by_task):
        selected.extend(sorted(by_task[task], key=lambda x: x["id"])[:per_task])
    return selected


def handoff(text: str, recent: str) -> str:
    return (
        "You are resuming a command-line task after context compaction. "
        "Prior history is below, followed by the recent history. Continue from "
        "the current terminal state and respond in the required Terminus JSON "
        "command format.\n\nPrior history:\n" + text +
        "\n\nRecent history:\n" + recent)


def prepare(args) -> None:
    import torch
    from llmlingua import PromptCompressor
    from transformers import AutoTokenizer

    inputs = [Path(x) for x in args.inputs]
    rows = select_rows(inputs, args.per_task)
    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL, trust_remote_code=True)
    compressor = PromptCompressor(model_name=LINGUA_MODEL, use_llmlingua2=True,
                                  device_map=args.device)
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        units = json.loads(row["units_json"])
        old = row["header"] + "".join(units)
        result = compressor.compress_prompt(old, rate=args.rate,
                                            force_tokens=["\n"])
        compressed = result["compressed_prompt"]
        old_ids = tok(old, add_special_tokens=False)["input_ids"]
        lingua_ids = tok(compressed, add_special_tokens=False)["input_ids"]
        matched_n = max(1, min(len(old_ids), len(lingua_ids)))
        recent = row["recent_text"]
        paths = list(dict.fromkeys(PATH_RE.findall(old)))
        path_hits = sum(p in compressed for p in paths)
        rec = {
            "id": row["id"], "task": row["task"],
            "source_model": row["source_model"],
            "logged_action": row["logged_action"],
            "old_qwen_tokens": len(old_ids),
            "llmlingua_qwen_tokens": len(lingua_ids),
            "achieved_rate": len(lingua_ids) / max(1, len(old_ids)),
            "path_total": len(paths), "path_exact_hits": path_hits,
            "path_exact_recall": path_hits / max(1, len(paths)),
            "llmlingua_report": {k: v for k, v in result.items()
                                  if k not in ("compressed_prompt", "compressed_prompt_list")},
            "prompts": {
                "full": handoff(old, recent),
                "keep_recent_matched": handoff(
                    tok.decode(old_ids[-matched_n:], skip_special_tokens=False), recent),
                "llmlingua2": handoff(compressed, recent),
            },
        }
        records.append(rec)
        output.write_text("\n".join(json.dumps(x) for x in records) + "\n")
        print(f"prepared {i+1}/{len(rows)} {row['task']} "
              f"rate={rec['achieved_rate']:.3f} paths={path_hits}/{len(paths)}")
    meta = {"experiment": "exp25_llmlingua2", "disposition": "exploratory",
            "compressor": LINGUA_MODEL, "compressor_version": "llmlingua==0.2.2",
            "source_model": SOURCE_MODEL, "rate_requested": args.rate,
            "per_task": args.per_task, "examples": len(records),
            "input_sha256": {str(p): sha256(p) for p in inputs}}
    output.with_suffix(".manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


def post_with_retry(url: str, payload: dict, attempts=3) -> dict:
    import requests
    error = None
    for i in range(attempts):
        try:
            r = requests.post(url, json=payload, timeout=1800)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            error = exc
            if i + 1 < attempts:
                time.sleep(10 * (i + 1))
    raise error


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def score(args) -> None:
    source = Path(args.prepared)
    rows = [json.loads(x) for x in source.open()]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = []
    if out.exists():
        done = [json.loads(x) for x in out.open()]
    completed = {(x["id"], x["condition"]) for x in done}
    url = args.executor_url.rstrip("/") + "/v1/chat/completions"
    for row in rows:
        for condition in CONDITIONS:
            if (row["id"], condition) in completed:
                continue
            payload = {
                "model": args.executor_name,
                "messages": [{"role": "user",
                              "content": row["prompts"][condition]}],
                "n": args.samples, "max_tokens": args.max_tokens,
                "temperature": 1.0, "top_p": 1.0,
                "reasoning_effort": "low",
            }
            response = post_with_retry(url, payload)
            texts = [c["message"].get("content") or ""
                     for c in response["choices"]]
            actions = [parse_action(x) for x in texts]
            logged = row["logged_action"]
            tool = lambda a: a.split("::", 1)[0] if a else None
            record = {
                "id": row["id"], "task": row["task"], "condition": condition,
                "logged_action": logged, "actions": actions,
                "acting": mean([a is not None for a in actions]),
                "tool_agreement": mean([tool(a) == tool(logged) for a in actions]),
                "verb_agreement": mean([_verb(a) == _verb(logged) for a in actions]),
                "exact_agreement": mean([a == logged for a in actions]),
                "achieved_rate": row["achieved_rate"],
                "path_exact_recall": row["path_exact_recall"],
            }
            done.append(record)
            with out.open("w") as f:
                for x in done: f.write(json.dumps(x) + "\n")
            print(f"scored {row['task']} {condition} tool={record['tool_agreement']:.2f}")

    grouped = {c: [x for x in done if x["condition"] == c] for c in CONDITIONS}
    metrics = ("acting", "tool_agreement", "verb_agreement", "exact_agreement")
    summary = {
        "experiment": "exp25_llmlingua2", "disposition": "exploratory",
        "source_model": SOURCE_MODEL, "compressor": LINGUA_MODEL,
        "examples": len(rows), "samples_per_condition": args.samples,
        "conditions": {c: {m: mean([x[m] for x in grouped[c]]) for m in metrics}
                       for c in CONDITIONS},
        "llmlingua": {
            "mean_achieved_rate": mean([x["achieved_rate"] for x in rows]),
            "mean_path_exact_recall": mean([x["path_exact_recall"] for x in rows]),
        },
        "prepared_sha256": sha256(source),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rate", type=float, default=0.25)
    p.add_argument("--per-task", type=int, default=2)
    p.add_argument("--device", default="cuda")
    s = sub.add_parser("score")
    s.add_argument("--prepared", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--summary", required=True)
    s.add_argument("--executor-url", default="http://127.0.0.1:8002")
    s.add_argument("--executor-name", default="qwen38-exp25")
    s.add_argument("--samples", type=int, default=4)
    s.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    (prepare if args.stage == "prepare" else score)(args)


if __name__ == "__main__":
    main()
