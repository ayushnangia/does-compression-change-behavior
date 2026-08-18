"""Pure data/rendering helpers for the exp24 extractive selector."""

from __future__ import annotations

import hashlib
import json
import re

TURN_RE = re.compile(r"<turn\b[^>]*>.*?</turn>\s*", re.DOTALL)


def stable_split(task: str) -> str:
    """Deterministic task-level 70/15/15 split."""
    x = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) % 100
    return "train" if x < 70 else ("validation" if x < 85 else "test")


def extract_turns(text: str) -> tuple[str, list[str]]:
    """Return exact pre-turn header and complete serialized turns."""
    matches = list(TURN_RE.finditer(text))
    if not matches:
        return "", [text] if text else []
    header = text[:matches[0].start()]
    # A capped context can start halfway through a turn. That partial prefix
    # is neither structural metadata nor free budget: drop it. Preserve only
    # a genuine complete trace header.
    if not header.lstrip().startswith("<agent_trace>"):
        header = ""
    return header, [m.group(0) for m in matches]


def parse_keep(completion: str, n_units: int) -> tuple[list[int], bool]:
    """Parse {\"keep\":[...]} strictly; return sorted unique valid indices."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(completion):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(completion[start:])
            raw = obj["keep"]
            if set(obj) != {"keep"} or not isinstance(raw, list):
                continue
            if any(type(i) is not int for i in raw):
                continue
            if any(i < 0 or i >= n_units for i in raw):
                continue
            return sorted(set(raw)), True
        except (ValueError, KeyError, TypeError):
            continue
    return [], False


def render_selection(
    header: str, units: list[str], keep: list[int], recent_text: str
) -> str:
    """Copy selected units and recent suffix byte-for-byte; never rewrite."""
    return header + "".join(units[i] for i in keep) + recent_text


def make_selector_prompt(units: list[str], budget_chars: int, max_chars=24000) -> str:
    """Compact manifest for the selector model. Rendering still uses full units."""
    intro = (
        "Select old agent-history blocks to preserve. The system will copy the "
        "selected blocks byte-for-byte and append the recent history. Max copied "
        f"characters: {budget_chars}. Prefer task state, commands, errors, paths, "
        "and unresolved decisions. Return JSON only: {\"keep\":[0,3,...]}\n\n"
    )
    remaining = max(0, max_chars - len(intro))
    per = max(160, remaining // max(1, len(units)))
    lines = []
    for i, unit in enumerate(units):
        preview = " ".join(unit.split())[:per]
        lines.append(f"[{i}] chars={len(unit)} {preview}")
    body = "\n".join(lines)
    return intro + body[:remaining]
