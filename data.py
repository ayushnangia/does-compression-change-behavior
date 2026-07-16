"""Load real agent traces and cut them at decision points.

A "decision point" is right before the agent calls a tool. We cut the history
there so that "what does the agent do next?" has a well-defined answer we can
compare across compressors.

Data: nvidia/Open-SWE-Traces (real software-agent runs). We stream it, so
nothing large is downloaded up front.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field


@dataclass
class Example:
    context_ids: "list[int]"   # the history up to a decision point, as token ids
    recent_ids: "list[int]"    # the most-recent slice, always kept raw
    repo: str                  # which codebase (used to keep train/test separate)
    logged_action: "str | None" = None  # the action the ORIGINAL agent took at
                                        # this decision point (parsed from the
                                        # trace) — an external ground truth
    future_segments: "list[list[int]]" = field(default_factory=list)
    # token ids of the trace BETWEEN subsequent decision points (each segment
    # ends right before the next tool call) — lets experiments teacher-force
    # the real continuation and measure how compression damage propagates


def _serialize(row: dict) -> str:
    """Turn a trace row into plain text with its turn/tool tags preserved."""
    parts = ["<agent_trace>"]
    for key in ("instance_id", "repo", "resolved"):
        if row.get(key) is not None:
            parts.append(f"<{key}>{row[key]}</{key}>")
    turns = row.get("trajectory") or row.get("messages") or []
    for i, msg in enumerate(turns if isinstance(turns, list) else []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("type") or "event"
        parts.append(f"<turn index={i} role={role}>")
        for fld in ("content", "text", "thought", "action", "observation"):
            if msg.get(fld):
                parts.append(f"<{fld}>\n{msg[fld]}\n</{fld}>")
        for fld in ("tool_calls", "tool_call", "function"):
            if msg.get(fld):
                parts.append(f"<{fld}>\n{msg[fld]}\n</{fld}>")
        parts.append("</turn>")
    parts.append("</agent_trace>")
    return "\n".join(parts)


def load_examples(
    tokenizer,
    *,
    context_tokens: int = 1024,
    recent_tokens: int = 128,
    num_examples: int = 8,
    max_per_repo: int = 4,
    skip_repos: "set[str] | None" = None,
    seed: int = 0,
) -> "list[Example]":
    """Return `num_examples` decision-point windows.

    `skip_repos` lets you hold out codebases (pass the training repos here so
    your test examples come from code the model's compressor never saw).
    """
    from datasets import load_dataset

    ds = load_dataset("nvidia/Open-SWE-Traces", "openhands", split="qwen35_122b", streaming=True)
    skip_repos = skip_repos or set()
    per_repo: dict[str, int] = {}
    out: list[Example] = []

    for row_idx, row in enumerate(ds):
        if len(out) >= num_examples:
            break
        repo = str(row.get("repo") or "unknown")
        if repo in skip_repos or per_repo.get(repo, 0) >= max_per_repo:
            continue

        text = _serialize(row)
        # decision points = positions right before a tool call opens
        anchors = [m.start() for m in re.finditer(r"<tool_calls>", text)]
        if not anchors:
            continue
        rng = random.Random(seed * 100003 + row_idx)
        rng.shuffle(anchors)

        for a in anchors:
            if len(out) >= num_examples or per_repo.get(repo, 0) >= max_per_repo:
                break
            ids = tokenizer(text[:a], add_special_tokens=False)["input_ids"]
            if len(ids) < context_tokens:
                continue
            window = ids[-context_tokens:]
            # the trace's own next action starts at the anchor — parse it with
            # the same parser used on model output, so labels are comparable
            from behavior import parse_action
            logged = parse_action(text[a:a + 2000])
            # teacher-forcing material: segments to the next 3 decision points
            futures = []
            later = sorted(x for x in anchors if x > a)[:3]
            bounds = [a] + later
            for i in range(len(later)):
                seg = tokenizer(text[bounds[i]:bounds[i + 1]],
                                add_special_tokens=False)["input_ids"]
                if len(seg) > 1536:   # keep contexts bounded; stop at a big gap
                    break
                futures.append(seg)
            out.append(Example(
                context_ids=window,
                recent_ids=window[-recent_tokens:],
                repo=repo,
                logged_action=logged,
                future_segments=futures,
            ))
            per_repo[repo] = per_repo.get(repo, 0) + 1

    if len(out) < num_examples:
        raise RuntimeError(f"only found {len(out)}/{num_examples} decision points")
    return out


def save_examples(examples: "list[Example]", path: str) -> None:
    """Serialize examples to JSON so offline compute nodes (no internet, e.g.
    Alliance clusters) can run without streaming from HuggingFace."""
    with open(path, "w") as f:
        json.dump([asdict(ex) for ex in examples], f)


def load_examples_file(path: str, num_examples: "int | None" = None) -> "list[Example]":
    """Load examples produced by save_examples (see prefetch.py)."""
    with open(path) as f:
        rows = json.load(f)
    if num_examples is not None:
        rows = rows[:num_examples]
    return [Example(**r) for r in rows]
