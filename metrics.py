"""Turn lists of actions into the three numbers we care about.

All three are simple and reported honestly:
- acting_rate: fraction of samples that are a real action (not None).
- action_change: how different two action lists are, 0 (identical mix) to 1
  (no overlap). This is total-variation distance between the two action
  distributions — but read it as "how much the behavior shifted."
- The floor: action_change between two runs of the SAME full context. Any
  compressor's change should be read RELATIVE to this, because random
  sampling alone makes even identical contexts differ a bit.
"""

from __future__ import annotations


def _label(a):
    return a if a is not None else "NO_ACTION"


def acting_rate(actions) -> float:
    """Fraction of samples where the model actually took an action."""
    return sum(a is not None for a in actions) / len(actions)


def lookup_rate(actions, action_kind) -> float:
    """Fraction of samples where the model chose an information-gathering
    (lookup) action rather than a task action or nothing."""
    return sum(action_kind(a) == "lookup" for a in actions) / len(actions)


def action_change(actions_a, actions_b) -> float:
    """0 = same mix of actions, 1 = completely different. (Total-variation
    distance between the two empirical action distributions.)"""
    la = [_label(a) for a in actions_a]
    lb = [_label(a) for a in actions_b]
    keys = set(la) | set(lb)
    pa = {k: la.count(k) / len(la) for k in keys}
    pb = {k: lb.count(k) / len(lb) for k in keys}
    return 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in keys)


def normalized_change(change: float, floor: float, ceiling: float = 1.0) -> float:
    """Rescale so 0 = the sampling floor and 1 = total change. Negative means
    'within noise'. This is what makes the number comparable across setups."""
    if ceiling <= floor:
        return float("nan")
    return (change - floor) / (ceiling - floor)
