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

import json
import math
from collections import Counter


def _label(a):
    return a if a is not None else "NO_ACTION"


def action_entropy(actions) -> float:
    """Shannon entropy (bits) of the sampled action distribution, NO_ACTION
    included. Mean behavior change hides HOW uncertainty is damaged:
    compression can collapse the distribution (overconfident on one action)
    or inflate it (scattered). Read against the full-context entropy —
    preserving the model's uncertainty structure, not just its argmax, is
    part of behavior preservation."""
    n = len(actions)
    return -sum((c / n) * math.log2(c / n)
                for c in Counter(_label(a) for a in actions).values())


def acting_rate(actions) -> float:
    """Fraction of samples where the model actually took an action."""
    return sum(a is not None for a in actions) / len(actions)


def lookup_rate(actions, action_kind) -> float:
    """Fraction of samples where the model chose an information-gathering
    (lookup) action rather than a task action or nothing."""
    return sum(action_kind(a) == "lookup" for a in actions) / len(actions)


def action_change_tools(actions_a, actions_b) -> float:
    """action_change at TOOL-NAME granularity (args stripped). Full labels
    saturate near 1.0 (ceiling); tool-level TV has a lower ceiling and
    detects compounding that label-level TV can hide."""
    strip = lambda a: a.split("::", 1)[0] if a else None
    return action_change([strip(a) for a in actions_a],
                         [strip(b) for b in actions_b])


def _verb(a):
    """Verb-level label: tool name + the first token of a shell command when
    present (bash_command -> 'bash_command ls'). A principled granularity
    between tool-only (too coarse) and full args (saturates): commands with
    the same verb are usually semantically close; different verbs rarely are."""
    if a is None:
        return None
    name, _, args = a.partition("::")
    import re as _re
    m = _re.search(r'(?:command|keystrokes)\\?":?\s*\\?"?\s*([A-Za-z0-9_./-]+)', args)
    return f"{name} {m.group(1)}" if m else name


def action_change_verbs(actions_a, actions_b) -> float:
    """action_change at verb granularity (see _verb)."""
    return action_change([_verb(a) for a in actions_a],
                         [_verb(b) for b in actions_b])


def ast_label(a):
    """Normalize an action label per the BFCL convention (Berkeley
    Function-Calling Leaderboard, Yan et al. 2024): a call is identified by
    function name + argument STRUCTURE (parsed, key-sorted), not by the
    surface string. 'edit::{"path": "a.py", "n": 1}' and
    'edit::{"n":1,"path":"a.py"}' are the same call; whitespace, key order
    and quoting style do not count as behavior change."""
    if a is None:
        return None
    name, _, args = a.partition("::")
    if not args:
        return name
    for loader in (json.loads,):
        try:
            obj = loader(args)
            if isinstance(obj, dict):
                canon = json.dumps(obj, sort_keys=True, separators=(",", ":"))
                return f"{name}::{canon}"
        except Exception:
            pass
    return f"{name}::{' '.join(args.split())}"  # non-JSON args: whitespace-normalized


def action_change_ast(actions_a, actions_b) -> float:
    """action_change at exact granularity under BFCL AST-equality."""
    return action_change([ast_label(a) for a in actions_a],
                         [ast_label(b) for b in actions_b])


def action_change_all(actions_a, actions_b) -> dict:
    """All three granularities at once. Reporting standard: quote all three;
    conclusions that hold at only one granularity are granularity artifacts."""
    return {"label": action_change(actions_a, actions_b),
            "verb": action_change_verbs(actions_a, actions_b),
            "tool": action_change_tools(actions_a, actions_b)}


def debiased_change(actions_a, actions_b, n_perm: int = 500, seed: int = 0):
    """Permutation-debiased divergence + exact p-value.

    Plug-in TV from 8 samples is inflated, and the inflation depends on the
    distributions' entropy, so a single shared floor is only approximately
    fair. Exact fix: pool the samples, permute the assignment, and use the
    permutation null PER COMPARISON.
      returns (excess, p) where excess = observed TV minus the null mean
      (a debiased effect size) and p is the exact one-sided p-value.
    """
    import random as _rd
    rng = _rd.Random(seed)
    obs = action_change(actions_a, actions_b)
    pool = list(actions_a) + list(actions_b)
    na = len(actions_a)
    null = []
    for _ in range(n_perm):
        rng.shuffle(pool)
        null.append(action_change(pool[:na], pool[na:]))
    null_mean = sum(null) / len(null)
    p = (1 + sum(v >= obs - 1e-12 for v in null)) / (n_perm + 1)
    return obs - null_mean, p


def harm_score(full_actions, comp_actions, logged: "str | None") -> dict:
    """Asymmetric divergence: counts only HARMFUL movement, so beneficial
    divergence scores zero (plain D penalizes improvements over the full-
    context reference, e.g. exp4's reasoning-deletion raising acting rate).
      halt_increase: new probability mass on NO_ACTION
      agree_drop:    lost agreement with the logged real action (if known)
    """
    p_no_f = sum(a is None for a in full_actions) / len(full_actions)
    p_no_c = sum(a is None for a in comp_actions) / len(comp_actions)
    out = {"halt_increase": max(0.0, p_no_c - p_no_f)}
    if logged is not None:
        lt = logged.split("::", 1)[0]
        ag = lambda acts: sum(a is not None and a.split("::", 1)[0] == lt
                              for a in acts) / len(acts)
        out["agree_drop"] = max(0.0, ag(full_actions) - ag(comp_actions))
    return out


def ngram_containment(derived_text: str, source_text: str, n: int = 3) -> float:
    """Fraction of the derived text's word n-grams that appear in the source
    (Zhang & Khattab 2026's input-side containment, applied to compaction:
    derived = compressed context, source = original context).
    1.0 = fully extractive (keep_recent), low = abstractive/novel text
    (LLM summaries). An INPUT-side measure to pair with output-side D."""
    from collections import Counter
    def grams(t):
        w = t.split()
        return Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    gd, gs = grams(derived_text), grams(source_text)
    if not gd:
        return 0.0
    inter = sum(min(c, gs.get(g, 0)) for g, c in gd.items())
    return inter / sum(gd.values())


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
