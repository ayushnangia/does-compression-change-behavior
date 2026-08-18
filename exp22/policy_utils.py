"""Pure, dependency-free helpers for exp22 live compaction policies."""

from __future__ import annotations


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
