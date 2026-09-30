"""Held-out evaluation for powered exp24 extractive selectors.

Stage 1 queries frozen base/LoRA Qwen3.5 selectors and constructs fixed
extractive controls. Stage 2 scores every valid selection with the same frozen
Qwen3.8 native-chat behavioral reward used during training. Validation chooses
one GRPO seed; the test split remains untouched until that adapter is frozen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import threading
import time

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from behavior import parse_action
from metrics import _verb
from experiments.exp24_credit import selector_reward
from experiments.exp24_data import make_selector_prompt, parse_keep, render_selection

SOURCE_MODEL = "Qwen/Qwen3.8-27B"
SELECTOR_MODELS = {
    "base_qwen35": "base_qwen35",
    "grpo_seed42": "grpo_seed42",
    "grpo_seed43": "grpo_seed43",
}
FIXED = ("keep_recent", "raw_skeleton")
RANDOM = ("random_matched_seed42", "random_matched_seed43")
CONDITIONS = tuple(SELECTOR_MODELS) + FIXED + RANDOM


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    if not rows:
        raise SystemExit("empty evaluation split")
    if {r.get("source_model") for r in rows} != {SOURCE_MODEL}:
        raise SystemExit("OFF-POLICY EVALUATION REFUSED")
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate evaluation row ids")
    return rows


def post_json(url: str, payload: dict, attempts: int = 3) -> dict:
    import requests
    error = None
    for attempt in range(attempts):
        try:
            response = requests.post(url, json=payload, timeout=1800)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(10 * (attempt + 1))
    raise error


def selected_chars(units: list[str], keep: list[int]) -> int:
    return sum(len(units[i]) for i in keep)


def keep_recent(units: list[str], budget: int) -> list[int]:
    keep, used = [], 0
    for i in range(len(units) - 1, -1, -1):
        cost = len(units[i])
        if used + cost <= budget:
            keep.append(i); used += cost
    return sorted(keep)


def raw_skeleton(units: list[str], budget: int) -> list[int]:
    """Exact command-bearing blocks plus exact newest blocks, no rewriting."""
    action = [i for i, unit in enumerate(units)
              if "<tool_calls>" in unit or parse_action(unit) is not None]
    keep, used = [], 0
    for i in reversed(action):
        if used + len(units[i]) <= budget // 2:
            keep.append(i); used += len(units[i])
    for i in range(len(units) - 1, -1, -1):
        if i not in keep and used + len(units[i]) <= budget:
            keep.append(i); used += len(units[i])
    return sorted(keep)


def random_matched(units: list[str], reference: list[int], budget: int,
                   key: str) -> list[int]:
    """Deterministic random control matched on cardinality and near token cost."""
    k = len(reference)
    if not k:
        return []
    target = selected_chars(units, reference)
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    indices = list(range(len(units)))
    best = None
    # Rejection sampling is unbiased over feasible k-subsets; retain the
    # closest-cost feasible draw to reduce a residual budget-size confound.
    for _ in range(5000):
        candidate = sorted(rng.sample(indices, k))
        cost = selected_chars(units, candidate)
        if cost <= budget:
            score = abs(cost - target)
            if best is None or score < best[0]:
                best = (score, candidate)
                if score == 0:
                    break
    if best is not None:
        return best[1]
    # A feasible set exists (the reference), but pathological heterogeneous
    # blocks can make rejection inefficient. Rotate the reference indices to
    # produce a deterministic different set when possible; otherwise return it.
    for shift in range(1, len(units)):
        candidate = sorted({(i + shift) % len(units) for i in reference})
        if len(candidate) == k and selected_chars(units, candidate) <= budget:
            return candidate
    return list(reference)


def append_jsonl(path: Path, record: dict, lock: threading.Lock) -> None:
    with lock, path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def selector_record(row: dict, condition: str, keep: list[int], valid: bool,
                    completion: str | None) -> dict:
    units = json.loads(row["units_json"])
    chars = selected_chars(units, keep) if valid else 0
    budget = int(row["budget_chars"])
    return {
        "id": row["id"], "task": row["task"], "condition": condition,
        "keep": keep, "completion": completion, "format_valid": valid,
        "selected_chars": chars, "budget_chars": budget,
        "budget_ratio": chars / max(1, budget),
        "within_budget": valid and chars <= budget,
    }


def select_stage(args) -> None:
    rows = load_rows(Path(args.split))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in out.open()] if out.exists() else []
    done = {(x["id"], x["condition"]) for x in existing}
    lock = threading.Lock()
    url = args.selector_url.rstrip("/") + "/v1/chat/completions"

    def query(row: dict, condition: str) -> dict:
        units = json.loads(row["units_json"])
        prompt = make_selector_prompt(units, int(row["budget_chars"]), 24000)
        payload = {
            "model": SELECTOR_MODELS[condition],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        result = post_json(url, payload)
        text = result["choices"][0]["message"].get("content") or ""
        keep, valid = parse_keep(text, len(units))
        return selector_record(row, condition, keep, valid, text)

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in rows:
            units = json.loads(row["units_json"]); budget = int(row["budget_chars"])
            for condition in SELECTOR_MODELS:
                if (row["id"], condition) not in done:
                    futures.append(pool.submit(query, row, condition))
            for condition, keep in (
                ("keep_recent", keep_recent(units, budget)),
                ("raw_skeleton", raw_skeleton(units, budget)),
            ):
                if (row["id"], condition) not in done:
                    rec = selector_record(row, condition, keep, True, None)
                    append_jsonl(out, rec, lock); done.add((row["id"], condition))
        for future in as_completed(futures):
            rec = future.result()
            append_jsonl(out, rec, lock); done.add((rec["id"], rec["condition"]))
            if len(done) % 50 == 0:
                print(f"selection progress {len(done)}/{len(rows)*len(CONDITIONS)}")

    # Random controls depend on the frozen learned selections.
    records = [json.loads(line) for line in out.open()]
    lookup = {(x["id"], x["condition"]): x for x in records}
    by_id = {r["id"]: r for r in rows}
    for row_id, row in by_id.items():
        units = json.loads(row["units_json"]); budget = int(row["budget_chars"])
        for seed in (42, 43):
            condition = f"random_matched_seed{seed}"
            if (row_id, condition) in done:
                continue
            ref = lookup[(row_id, f"grpo_seed{seed}")]
            if ref["format_valid"] and ref["within_budget"]:
                keep = random_matched(units, ref["keep"], budget,
                                      f"{row_id}:{seed}:exp24-random")
                rec = selector_record(row, condition, keep, True, None)
            else:
                # Match the deployed invalidity of the reference rather than
                # silently giving random a valid action it did not earn.
                rec = selector_record(row, condition, [], False, None)
            append_jsonl(out, rec, lock); done.add((row_id, condition))
    expected = len(rows) * len(CONDITIONS)
    if len(done) != expected:
        raise SystemExit(f"selection rows incomplete: {len(done)} != {expected}")
    print(f"selection stage complete: {expected} rows -> {out}")


def score_stage(args) -> None:
    source_rows = load_rows(Path(args.split))
    rows = {r["id"]: r for r in source_rows}
    selections = [json.loads(line) for line in Path(args.selections).open()]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in out.open()] if out.exists() else []
    done = {(x["id"], x["condition"]) for x in existing}
    lock = threading.Lock()
    url = args.executor_url.rstrip("/") + "/v1/chat/completions"

    def score(selection: dict) -> dict:
        row = rows[selection["id"]]
        valid = selection["format_valid"] and selection["within_budget"]
        if not valid:
            reward = selector_reward(tool_agreement=0, verb_agreement=0,
                acting_rate=0, valid=selection["format_valid"],
                budget_ratio=selection["budget_ratio"])
            return {**selection, "reward": reward, "actions": [],
                    "tool_agreement": 0.0, "verb_agreement": 0.0,
                    "acting": 0.0, "executor_scored": False}
        units = json.loads(row["units_json"])
        context = render_selection(row["header"], units, selection["keep"],
                                   row["recent_text"])
        handoff = (
            "You are resuming a command-line task after context compaction. "
            "Selected prior trace blocks are copied below verbatim, followed "
            "by the recent history. Continue from the current terminal state "
            "and respond in the required Terminus JSON command format.\n\n" +
            context)
        payload = {
            "model": args.executor_name,
            "messages": [{"role": "user", "content": handoff}],
            "n": args.samples, "max_tokens": args.max_tokens,
            "temperature": 1.0, "top_p": 1.0, "reasoning_effort": "low",
        }
        result = post_json(url, payload)
        texts = [x["message"].get("content") or "" for x in result["choices"]]
        actions = [parse_action(text) for text in texts]
        logged = row["logged_action"]
        tool = lambda a: a.split("::", 1)[0] if a else None
        acting = sum(a is not None for a in actions) / len(actions)
        tool_agree = sum(tool(a) == tool(logged) for a in actions) / len(actions)
        verb_agree = sum(_verb(a) == _verb(logged) for a in actions) / len(actions)
        reward = selector_reward(tool_agreement=tool_agree,
            verb_agreement=verb_agree, acting_rate=acting, valid=True,
            budget_ratio=selection["budget_ratio"])
        return {**selection, "reward": reward, "actions": actions,
                "tool_agreement": tool_agree, "verb_agreement": verb_agree,
                "acting": acting, "executor_scored": True}

    pending = [x for x in selections if (x["id"], x["condition"]) not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score, selection) for selection in pending]
        for future in as_completed(futures):
            rec = future.result(); append_jsonl(out, rec, lock)
            done.add((rec["id"], rec["condition"]))
            if len(done) % 50 == 0:
                print(f"score progress {len(done)}/{len(selections)}")
    if len(done) != len(selections):
        raise SystemExit(f"score rows incomplete: {len(done)} != {len(selections)}")
    summarize(Path(args.split), Path(args.selections), out, Path(args.summary), args)


def bootstrap_diff(a: list[float], b: list[float], seed: int = 24) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed); n = len(diffs)
    draws = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n
             for _ in range(10000)]
    draws.sort()
    return {"mean": statistics.mean(diffs), "ci95": [draws[249], draws[9749]]}


def summarize(split: Path, selections_path: Path, scores_path: Path,
              summary_path: Path, args) -> None:
    rows = [json.loads(line) for line in scores_path.open()]
    by_condition = {c: sorted((x for x in rows if x["condition"] == c),
                              key=lambda x: x["id"]) for c in CONDITIONS}
    metrics = {}
    for condition, xs in by_condition.items():
        scored = [x for x in xs if x["executor_scored"]]
        metrics[condition] = {
            "n": len(xs),
            "mean_reward": statistics.mean(x["reward"] for x in xs),
            "format_valid_rate": statistics.mean(x["format_valid"] for x in xs),
            "within_budget_rate": statistics.mean(x["within_budget"] for x in xs),
            "mean_budget_ratio": statistics.mean(x["budget_ratio"] for x in xs),
            "mean_kept_blocks": statistics.mean(len(x["keep"]) for x in xs),
            "executor_scored_n": len(scored),
            "clean_tool_agreement": statistics.mean(x["tool_agreement"] for x in scored) if scored else None,
            "clean_verb_agreement": statistics.mean(x["verb_agreement"] for x in scored) if scored else None,
            "clean_acting": statistics.mean(x["acting"] for x in scored) if scored else None,
            "task_mean_reward": {task: statistics.mean(x["reward"] for x in xs if x["task"] == task)
                                 for task in sorted({x["task"] for x in xs})},
        }
    paired = {}
    for seed in (42, 43):
        condition = f"grpo_seed{seed}"
        for control in ("base_qwen35", "keep_recent", f"random_matched_seed{seed}"):
            paired[f"{condition}_minus_{control}"] = bootstrap_diff(
                [x["reward"] for x in by_condition[condition]],
                [x["reward"] for x in by_condition[control]], seed=2400 + seed)
    winner = max(("grpo_seed42", "grpo_seed43"),
                 key=lambda c: (metrics[c]["mean_reward"], c == "grpo_seed42"))
    adapters = {
        "grpo_seed42": Path(args.adapter42),
        "grpo_seed43": Path(args.adapter43),
    }
    chosen_weight = adapters[winner] / "adapter_model.safetensors"
    summary = {
        "experiment": "exp24_powered_validation",
        "disposition": "validation selects seed; test remains untouched",
        "source_model": SOURCE_MODEL,
        "split": str(split), "split_sha256": sha256(split),
        "examples": len(by_condition[winner]), "tasks": sorted({x["task"] for x in rows}),
        "executor_samples": args.samples,
        "conditions": metrics, "paired_row_bootstrap_descriptive": paired,
        "selection_rule": "higher validation mean direct reward; exact tie -> seed42",
        "selected_condition": winner,
        "selected_adapter": str(adapters[winner]),
        "selected_adapter_sha256": sha256(chosen_weight),
        "adapter_sha256": {c: sha256(p / "adapter_model.safetensors")
                           for c, p in adapters.items()},
        "selections_sha256": sha256(selections_path),
        "scores_sha256": sha256(scores_path),
        "limitations": [
            "Validation has two held-out tasks; row bootstrap is descriptive, not task-level evidence.",
            "The untouched test split is not read by this job.",
            "Training-curve reward is not used for seed selection.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--split", required=True)
    common.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("select", parents=[common])
    p.add_argument("--selector-url", required=True)
    p.add_argument("--out", required=True)
    s = sub.add_parser("score", parents=[common])
    s.add_argument("--selections", required=True)
    s.add_argument("--executor-url", required=True)
    s.add_argument("--executor-name", default="qwen38-exp24-eval")
    s.add_argument("--samples", type=int, default=4)
    s.add_argument("--max-tokens", type=int, default=4096)
    s.add_argument("--out", required=True)
    s.add_argument("--summary", required=True)
    s.add_argument("--adapter42", required=True)
    s.add_argument("--adapter43", required=True)
    args = ap.parse_args()
    if args.stage == "select": select_stage(args)
    else: score_stage(args)


if __name__ == "__main__":
    main()
