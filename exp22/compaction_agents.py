"""EXP 22 - Outcome-grounded compaction: policy arms as Terminus-2 subclasses.

Harbor's AgentFactory loads agents by import path, so these plug into a job
config with NO fork of harbor:

    agents:
      - import_path: exp22.compaction_agents:KeepRecentTerminus
        model_name: hosted_vllm/qwen35-35b
        kwargs: {...}

Arms:
  StockCappedTerminus   arm C: deployed 3-step summarizer (harbor native)
  KeepRecentTerminus    arm A: keep the newest messages verbatim, drop the rest
  RawSkeletonTerminus   arm B: byte-exact command-bearing messages + verbatim tail
  LearnedSelectorTerminus: trained Qwen3.5 index selector for the final learned arm
  (arm D, no compaction: stock Terminus2 with enable_summarize=False)

The policy arms replace ONLY _summarize(); triggering (proactive threshold,
context-limit) stays identical. A/B/C use their assigned policy for the first
three compactions, then the same keep-recent fallback. Temperature/top_p come
from the serving side (1.0/1.0); episodes allow 250 turns.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from harbor.agents.terminus_2.terminus_2 import Terminus2

from exp22.policy_utils import (fit_raw_skeleton, render_live_selection,
                                split_old_recent)
from experiments.exp24_data import make_selector_prompt, parse_keep


def _msg_text(m) -> str:
    c = getattr(m, "content", None)
    if c is None and isinstance(m, dict):
        c = m.get("content", "")
    return c if isinstance(c, str) else json.dumps(c, default=str)


def _extract_keystrokes(text: str) -> list[str]:
    """Extract command keystrokes from a Terminus response using HARBOR'S OWN
    parser (with its auto-fixes), not a homegrown regex - the action space is
    defined by the scaffold, so the scaffold's parser is the ground truth."""
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )
    result = TerminusJSONPlainParser().parse_response(text)
    return [c.keystrokes.strip() for c in result.commands if c.keystrokes.strip()]



MAX_COMPACTIONS = 3  # CompactionRL parity: "at most three compaction operations"
# ONE handoff budget shared by every arm (chars ~ 4 chars/token ~ 6k tokens).
# Arms may only differ in WHAT fills the budget, never in HOW MUCH - otherwise
# the comparison is budget-confounded.
HANDOFF_BUDGET_CHARS = 24000
# The trained contract always appends recent history outside selection. Live
# deployment reserves half of the shared handoff budget for that exact suffix;
# the selector's old-block budget is whatever remains. All content still fits
# the same total 24k-character envelope as the other exp22 arms.
LEARNED_RECENT_BUDGET_CHARS = HANDOFF_BUDGET_CHARS // 2


def _query_selector(prompt: str) -> str:
    """Call the local frozen Qwen3.5 selector through native chat."""
    base = os.environ.get("EXP24_SELECTOR_API_BASE", "http://127.0.0.1:8002/v1")
    model = os.environ.get("EXP24_SELECTOR_MODEL", "qwen35-exp24-selector")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    return result["choices"][0]["message"].get("content") or ""


def _log_selector(record: dict) -> None:
    path = os.environ.get("EXP24_SELECTOR_LOG")
    if path:
        # One compact append is safe across Harbor's low-concurrency workers;
        # evaluation also checks row counts/fallback rates before making claims.
        with open(path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _capped(self) -> bool:
    return self._summarization_count >= MAX_COMPACTIONS


def _fallback_keep_recent(msgs, original_instruction, budget_chars=HANDOFF_BUDGET_CHARS):
    kept, used = [], 0
    for m in reversed(msgs):
        t = _msg_text(m)
        if used + len(t) > budget_chars and kept:
            break
        kept.append(t); used += len(t)
    kept.reverse()
    return (f"Original task:\n{original_instruction}\n\n"
            "You are resuming this task. Older history was dropped; the most "
            "recent exchanges are below, verbatim.\n\n" + "\n\n".join(kept) +
            "\n\nContinue the task from the current terminal state.")


class KeepRecentTerminus(Terminus2):
    """Arm A: verbatim-recency compaction. Keeps the newest messages whole
    until ~half the proactive threshold is used; drops everything older."""

    _POLICY = "keep_recent"

    async def _summarize(self, chat, original_instruction, session):
        msgs = list(getattr(chat, "messages", []) or [])
        if not msgs:
            return original_instruction, None
        if _capped(self):  # parity cap: identical fallback across A/B/C
            self._summarization_count += 1
            return _fallback_keep_recent(msgs, original_instruction), None
        budget_chars = HANDOFF_BUDGET_CHARS
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


class RawSkeletonTerminus(Terminus2):
    """Arm B: exact command-bearing agent messages plus an exact recent tail.

    exp23 rejected canonical one-liners: rewriting lost about 13 agreement
    points while raw skeleton+tail tied keep-recent. Therefore this live arm
    performs no canonicalization or paraphrase. Command-bearing messages are
    identified with Harbor's parser but copied byte-for-byte.
    """

    _POLICY = "raw_skeleton"

    async def _summarize(self, chat, original_instruction, session):
        msgs = list(getattr(chat, "messages", []) or [])
        if not msgs:
            return original_instruction, None
        if _capped(self):  # parity cap: identical fallback across A/B/C
            self._summarization_count += 1
            return _fallback_keep_recent(msgs, original_instruction), None

        texts = [_msg_text(m) for m in msgs]
        action_mask = [bool(_extract_keystrokes(t)) for t in texts]
        skeleton, tail = fit_raw_skeleton(
            texts, action_mask, HANDOFF_BUDGET_CHARS
        )
        handoff = (
            f"Original task:\n{original_instruction}\n\n"
            "You are resuming this task. Older command-bearing responses and "
            "the recent history are copied below verbatim.\n\n"
            f"Command history:\n{skeleton}\n\n"
            f"Recent history:\n{tail}\n\n"
            "Continue the task from the current terminal state."
        )
        self._summarization_count += 1
        return handoff, None


class LearnedSelectorTerminus(Terminus2):
    """Our deployed Qwen3.5 LoRA selector on Qwen3.8's own live history.

    This is delegated self-compaction: the selector sees old blocks from the
    currently acting Qwen3.8 trajectory, emits only {"keep":[...]}, and this
    class copies those message bodies verbatim. Invalid output, overflow, or a
    dead selector endpoint falls back safely to the predeclared recency policy
    and is logged; fallback rates must be reported with outcomes.
    """

    _POLICY = "learned_qwen35_selector"

    async def _summarize(self, chat, original_instruction, session):
        msgs = list(getattr(chat, "messages", []) or [])
        if not msgs:
            return original_instruction, None
        if _capped(self):
            self._summarization_count += 1
            return _fallback_keep_recent(msgs, original_instruction), None

        texts = [_msg_text(m) for m in msgs]
        old, recent = split_old_recent(texts, LEARNED_RECENT_BUDGET_CHARS)
        old_budget = max(0, HANDOFF_BUDGET_CHARS - len(recent))
        record = {
            "policy": self._POLICY,
            "compaction": self._summarization_count,
            "old_blocks": len(old),
            "recent_chars": len(recent),
            "budget_chars": HANDOFF_BUDGET_CHARS,
            "selector_model": os.environ.get(
                "EXP24_SELECTOR_MODEL", "qwen35-exp24-selector"),
        }
        try:
            if old and old_budget:
                prompt = make_selector_prompt(old, old_budget, max_chars=24000)
                completion = await asyncio.to_thread(_query_selector, prompt)
                keep, valid = parse_keep(completion, len(old))
            else:
                completion, keep, valid = '{"keep":[]}', [], True
            body, within_budget = render_live_selection(
                old, keep, recent, HANDOFF_BUDGET_CHARS)
            valid = valid and within_budget
            record.update({"completion": completion, "keep": keep,
                           "valid": valid, "selected_chars": len(body)})
        except Exception as exc:
            valid = False
            record.update({"valid": False, "error":
                           f"{type(exc).__name__}: {exc}"})

        if not valid:
            record["fallback"] = "keep_recent"
            handoff = _fallback_keep_recent(msgs, original_instruction)
        else:
            record["fallback"] = None
            handoff = (
                f"Original task:\n{original_instruction}\n\n"
                "You are resuming this task after learned context compaction. "
                "The selected older messages and recent history below are "
                "copied verbatim.\n\n" + body +
                "\n\nContinue the task from the current terminal state."
            )
        _log_selector(record)
        self._summarization_count += 1
        return handoff, None


class StockCappedTerminus(Terminus2):
    """Arm C: stock three-step summarization for three events, then the same
    fallback as A/B. This makes the cap policy identical across live arms."""

    _POLICY = "stock_summary"

    async def _summarize(self, chat, original_instruction, session):
        msgs = list(getattr(chat, "messages", []) or [])
        if not msgs:
            return original_instruction, None
        if _capped(self):
            self._summarization_count += 1
            return _fallback_keep_recent(msgs, original_instruction), None
        return await super()._summarize(chat, original_instruction, session)


# Compatibility only for old smoke configs; never use in paper runs.
OneLinerTerminus = RawSkeletonTerminus
