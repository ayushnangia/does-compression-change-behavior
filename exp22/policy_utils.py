"""Pure, dependency-free helpers for exp22 live compaction policies."""

from __future__ import annotations


def split_old_recent(
    texts: list[str], recent_budget_chars: int
) -> tuple[list[str], str]:
    """Reserve a newest complete-message suffix for learned compaction.

    Training always appends a recent suffix outside the selector. Live use
    mirrors that contract while keeping the total handoff budget explicit. If
    one newest message alone exceeds the reserve, retain only its byte-exact
    suffix; there are then no selectable old blocks.
    """
    if recent_budget_chars < 0:
        raise ValueError("recent budget must be nonnegative")
    if not texts or recent_budget_chars == 0:
        return list(texts), ""
    kept: list[str] = []
    used = 0
    split = len(texts)
    for i in range(len(texts) - 1, -1, -1):
        text = texts[i]
        cost = len(text) + (2 if kept else 0)
        if used + cost > recent_budget_chars:
            break
        kept.append(text)
        used += cost
        split = i
    if not kept:
        # A pathological giant terminal observation must not silently violate
        # the shared budget. Keep a literal suffix, never a paraphrase.
        return list(texts[:-1]), texts[-1][-recent_budget_chars:]
    kept.reverse()
    return list(texts[:split]), "\n\n".join(kept)


def render_live_selection(
    old_texts: list[str], keep: list[int], recent: str, budget_chars: int
) -> tuple[str, bool]:
    """Render selected message bodies verbatim and enforce total budget."""
    if any(type(i) is not int or i < 0 or i >= len(old_texts) for i in keep):
        return "", False
    selected = "\n\n".join(old_texts[i] for i in sorted(set(keep)))
    body = selected + (("\n\n" if selected and recent else "") + recent)
    return body, len(body) <= budget_chars


def fit_raw_skeleton(
    texts: list[str], action_mask: list[bool], budget_chars: int
) -> tuple[str, str]:
    """Fit exact action-bearing messages + exact recent suffix into one budget.

    At most half the budget is allocated to action-bearing messages. Newest
    complete action messages are preferred; the rest is filled by a verbatim
    suffix of the complete message serialization. No input text is rewritten.
    """
    if len(texts) != len(action_mask):
        raise ValueError("texts and action_mask must align")
    if budget_chars < 0:
        raise ValueError("budget_chars must be nonnegative")
    action_units = [t for t, acted in zip(texts, action_mask) if acted]
    skeleton_budget = budget_chars // 2
    kept, used = [], 0
    for text in reversed(action_units):
        cost = len(text) + (2 if kept else 0)
        if used + cost > skeleton_budget:
            break
        kept.append(text)
        used += cost
    kept.reverse()
    skeleton = "\n\n".join(kept)
    tail_budget = budget_chars - len(skeleton)
    serialized = "\n\n".join(texts)
    tail = serialized[-tail_budget:] if tail_budget else ""
    return skeleton, tail
