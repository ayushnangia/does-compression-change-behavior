"""EXP 22 - Outcome-grounded compaction: policy arms as Terminus-2 subclasses.

Harbor's AgentFactory loads agents by import path, so these plug into a job
config with NO fork of harbor:

    agents:
      - import_path: exp22.compaction_agents:KeepRecentTerminus
        model_name: hosted_vllm/qwen35-35b
        kwargs: {...}

Arms:
  Terminus2 (stock)     arm C: the deployed 3-step summarizer (harbor native)
  KeepRecentTerminus    arm A: keep the newest messages verbatim, drop the rest
  OneLinerTerminus      arm B: canonical one-liner action history (exp21),
                        wrapped in the model-native format
  (arm D, no compaction: stock Terminus2 with enable_summarize=False)

Both overrides replace ONLY _summarize(); triggering (proactive threshold,
context-limit) stays identical to stock, so the arms differ in policy alone.
CompactionRL parity: temperature/top_p come from the serving side (1.0/1.0),
<=3 compactions enforced via max_compactions, 250-turn episodes in the job
config.
"""

from __future__ import annotations

import json
import re

from harbor.agents.terminus_2.terminus_2 import Terminus2

# canonicalizer shared with exp21 (same repo)
READ_VERBS = {"cat", "less", "head", "tail", "view", "open"}


def _canon_command(cmd: str) -> str:
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


def _msg_text(m) -> str:
    c = getattr(m, "content", None)
    if c is None and isinstance(m, dict):
        c = m.get("content", "")
    return c if isinstance(c, str) else json.dumps(c, default=str)


def _extract_keystrokes(text: str) -> list[str]:
    """Pull command keystrokes out of a Terminus JSON response (tolerant)."""
    out = []
    for m in re.finditer(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        try:
            out.append(json.loads(f'"{m.group(1)}"').strip())
        except Exception:
            out.append(m.group(1).strip())
    return [k for k in out if k]


class KeepRecentTerminus(Terminus2):
    """Arm A: verbatim-recency compaction. Keeps the newest messages whole
    until ~half the proactive threshold is used; drops everything older."""

    _POLICY = "keep_recent"

    async def _summarize(self, chat, original_instruction, session):
        msgs = list(getattr(chat, "messages", []) or [])
        if not msgs:
            return original_instruction, None
        budget_chars = 24000  # ~6k tokens of verbatim recency
        kept, used = [], 0
        for m in reversed(msgs):
            t = _msg_text(m)
            if used + len(t) > budget_chars and kept:
                break
            kept.append(t)
            used += len(t)
        kept.reverse()
        recent = "\n\n".join(kept)
        handoff = (
            f"Original task:\n{original_instruction}\n\n"
            "You are resuming this task. Older history was dropped; the most "
            "recent exchanges are below, verbatim.\n\n"
            f"{recent}\n\n"
            "Continue the task from the current terminal state."
        )
        self._summarization_count += 1
        return handoff, None


class OneLinerTerminus(Terminus2):
    """Arm B: canonical one-liner action history (exp21, wrapped delivery) +
    a small verbatim recent tail."""

    _POLICY = "one_liner"

    async def _summarize(self, chat, original_instruction, session):
        msgs = list(getattr(chat, "messages", []) or [])
        if not msgs:
            return original_instruction, None
        lines = []
        for m in msgs:
            for ks in _extract_keystrokes(_msg_text(m)):
                lines.append(_canon_command(ks))
        history = "\n".join(f"<tool_call>{l}</tool_call>" for l in lines[-200:])
        tail = "\n\n".join(_msg_text(m) for m in msgs[-4:])[:8000]
        handoff = (
            f"Original task:\n{original_instruction}\n\n"
            "You are resuming this task. Complete record of every command "
            "you have run so far, in order, in shorthand:\n\n"
            f"{history}\n\n"
            f"Most recent exchanges, verbatim:\n{tail}\n\n"
            "Continue the task from the current terminal state."
        )
        self._summarization_count += 1
        return handoff, None
