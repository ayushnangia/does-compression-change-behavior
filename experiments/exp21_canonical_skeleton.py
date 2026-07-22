"""EXP 21 (GPU) — Canonical action shorthand (group proposal): rewrite each
tool call as a compact canonical line (GREP pattern file / READ path /
EDIT path gist) instead of keeping the raw JSON block.

Rationale from our own findings: tool calls carry the behavior (exp4) but
each block is mostly JSON boilerplate; canonicalizing is ~10x on the
load-bearing tokens. Risk from our own findings: novel surface forms can
cost behavior (exp14/exp15 format effects) even at constant information.

Conditions at each rate (exp17-style extreme sweep):
  skeleton           raw <tool_calls> blocks, random-trimmed (exp17 baseline)
  canonical_bare     shorthand lines, no tags
  canonical_wrapped  the same shorthand inside <tool_calls>...</tool_calls>
  keep_recent        anchor

Pre-registered predictions (stated before running):
  P1 at moderate rates (12.5-25%): canonical ~= skeleton
  P2 at extreme rates (2-5%): canonical > skeleton (boilerplate crowds out
     calls in the raw form)
  P3 wrapped >= bare (format familiarity)

    python exp21_canonical_skeleton.py --examples-file ../examples_16k_large.json
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re

from common import load_model, save_result, REPO  # noqa: F401
from stats import fmt_ci, paired_permutation_p
from behavior import sample_actions
from data import load_examples_file
from exp4_block_ablation import random_trim
from metrics import acting_rate, action_change_tools

TOOLCALL_BLOCK = re.compile(r"<tool_calls>(.*?)</tool_calls>", re.DOTALL)
READ_VERBS = {"cat", "less", "head", "tail", "view", "open"}


def _parse_calls(body: str):
    """Best-effort parse of one block's calls -> list of (name, args_dict)."""
    try:
        obj = ast.literal_eval(body.strip())
    except Exception:
        try:
            obj = json.loads(body.strip())
        except Exception:
            return []
    if isinstance(obj, dict):
        obj = [obj]
    out = []
    for call in obj if isinstance(obj, list) else []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function", call)
        name = (fn.get("name") if isinstance(fn, dict) else None) \
            or call.get("name") or call.get("function_name") or "TOOL"
        raw = (fn.get("arguments") if isinstance(fn, dict) else None) \
            or call.get("arguments") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {"_": raw}
        out.append((str(name), raw if isinstance(raw, dict) else {"_": str(raw)}))
    return out


def _canon_command(cmd: str) -> str:
    """'cd /x && grep -n Parse f.go | head -3' -> 'CD /x; GREP Parse f.go; HEAD'
    Flags dropped, verbs uppercased, read-alias normalized, tiered per the
    agreed rule (cheap verbs: verb+targets only)."""
    segs = re.split(r"\s*(?:&&|\|\||\||;)\s*", cmd.strip())
    outs = []
    for s in segs:
        toks = [t for t in s.split() if t]
        if not toks:
            continue
        verb = toks[0].lower()
        verb = "read" if verb in READ_VERBS else verb
        args = [t for t in toks[1:] if not t.startswith("-")][:3]
        outs.append((verb.upper() + " " + " ".join(args)).strip())
    return "; ".join(outs)[:120]


def canonicalize(name: str, args: dict) -> str:
    cmd = args.get("command") or args.get("keystrokes")
    if cmd:
        return _canon_command(str(cmd))
    path = args.get("path") or args.get("file") or args.get("file_path") or ""
    n = name.lower()
    if "write" in n or "create" in n:
        gist = str(args.get("file_text") or args.get("content") or "")[:40]
        return f"WRITE {path} {gist}".strip()
    if "replace" in n or "edit" in n:
        gist = str(args.get("new_str") or args.get("old_str") or "")[:40]
        return f"EDIT {path} {gist}".strip()
    if "view" in n or "read" in n or "open" in n:
        return f"READ {path}".strip()
    other = " ".join(str(v)[:30] for v in list(args.values())[:2])
    return f"{name.upper()} {other}".strip()


def build(cond, old_text, budget, tokenizer, rng):
    ids = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]
    blocks = TOOLCALL_BLOCK.findall(old_text)
    if cond == "keep_recent":
        return random_trim(ids(old_text), budget, rng)   # placeholder; replaced below
    if cond == "skeleton":
        raw = "\n".join(f"<tool_calls>{b}</tool_calls>" for b in blocks)
        return random_trim(ids(raw), budget, rng)
    lines = []
    for b in blocks:
        for name, args in _parse_calls(b):
            lines.append(canonicalize(name, args))
    if cond == "canonical_bare":
        text = "\n".join(lines)
    else:  # canonical_wrapped
        text = "\n".join(f"<tool_calls>{l}</tool_calls>" for l in lines)
    return random_trim(ids(text), budget, rng)


def agreement(actions, logged):
    if logged is None:
        return None
    lt = logged.split("::", 1)[0]
    return sum(a is not None and a.split("::", 1)[0] == lt
               for a in actions) / len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--num-examples", type=int, default=24)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.02, 0.05, 0.125, 0.25])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model)
    examples = load_examples_file(args.examples_file, args.num_examples)
    conds = ["keep_recent", "skeleton", "canonical_bare", "canonical_wrapped"]

    stats = {(c, r): {"acting": [], "coarse": [], "agree": [], "used": []}
             for c in conds for r in args.rates}
    floors = []
    usable = 0

    for ei, ex in enumerate(examples):
        base = args.seed * 9973 + ei * 17
        rng = random.Random(base)
        old = ex.context_ids[: -len(ex.recent_ids)]
        old_text = tokenizer.decode(old, skip_special_tokens=True)

        full_a = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 1)
        if acting_rate(full_a) < 0.5:
            print(f"example {ei}: skipped")
            continue
        usable += 1
        full_b = sample_actions(model, tokenizer, ex.context_ids, device,
                                samples=args.samples, seed=base + 2)
        floors.append(action_change_tools(full_a, full_b))

        for c in conds:
            for r in args.rates:
                budget = max(16, int(len(old) * r))
                if c == "keep_recent":
                    comp = old[-budget:]
                else:
                    comp = build(c, old_text, budget, tokenizer, rng)
                acts = sample_actions(model, tokenizer,
                                      list(comp) + list(ex.recent_ids), device,
                                      samples=args.samples, seed=base + 3)
                cell = stats[(c, r)]
                cell["acting"].append(acting_rate(acts))
                cell["coarse"].append(action_change_tools(full_a, acts))
                cell["used"].append(len(comp))
                ag = agreement(acts, ex.logged_action)
                if ag is not None:
                    cell["agree"].append(ag)
        print(f"example {ei}: done")

    if not usable:
        print("no usable examples")
        return
    floor = sum(floors) / len(floors)
    print(f"\nusable: {usable}   coarse floor: {floor:.3f}")
    print("cells: coarse-D | acting | agreement | avg tokens used\n")
    rows = {}
    for c in conds:
        cells = []
        for r in args.rates:
            s = stats[(c, r)]
            n = len(s["acting"])
            co, ac = sum(s["coarse"]) / n, sum(s["acting"]) / n
            ag = sum(s["agree"]) / len(s["agree"]) if s["agree"] else float("nan")
            uk = sum(s["used"]) / n
            cells.append(f" R={r:g}: {co:.2f}|{ac:.2f}|{ag:.2f}|{uk:.0f}t")
            rows[f"{c}@{r:g}"] = {"coarse": co, "acting": ac, "agree": ag,
                                  "tokens_used": uk}
        print(f"{c:18s}" + "".join(cells))
    # pre-registered primary endpoints
    r_lo = args.rates[0]
    for a, b, label in (("canonical_wrapped", "skeleton", "P2 (extreme rate)"),
                        ("canonical_wrapped", "canonical_bare", "P3 (format)")):
        p = paired_permutation_p(stats[(a, r_lo)]["agree"],
                                 stats[(b, r_lo)]["agree"])
        print(f"paired p (agreement @R={r_lo:g}, {a} vs {b}) [{label}]: {p:.3f}")

    save_result("exp21_canonical_skeleton", {
        "model": args.model, "usable": usable, "floor": floor,
        "rates": args.rates, "cells": rows,
        "raw": {f"{c}@{r:g}": stats[(c, r)] for c in conds for r in args.rates},
    })


if __name__ == "__main__":
    main()
