"""EXP 24A (CPU) — credit rule for direct-reward compactor GRPO.

Compaction is a persistent intervention, not an ordinary token action. We avoid
GAE entirely: at a fixed history, generate G candidate memories, score the
executor immediately, and normalize only within that history. Each trajectory
has total weight one, divided across its compaction events, so a rollout with
four compactions cannot receive four times the gradient mass.

This module is dependency-free and used by the training code and unit tests.
"""

from __future__ import annotations

import math


def group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Population-standardized rewards for candidates from ONE history."""
    if not rewards:
        raise ValueError("reward group cannot be empty")
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    sd = math.sqrt(var)
    if sd < eps:
        return [0.0] * len(rewards)
    return [(r - mean) / sd for r in rewards]


def event_weights(num_compactions: int) -> list[float]:
    """Equal persistent-intervention credit with unit trajectory mass."""
    if num_compactions < 1:
        raise ValueError("num_compactions must be positive")
    return [1.0 / num_compactions] * num_compactions


def selector_reward(
    *, tool_agreement: float, verb_agreement: float, acting_rate: float,
    valid: bool, budget_ratio: float,
) -> float:
    """Frozen direct reward used by exp24.

    Agreement is external to the compactor. Acting is a small tie-breaker, not
    the objective. Invalid JSON and budget overflow are hard failures.
    """
    if not valid or budget_ratio > 1.0:
        return -1.0 - max(0.0, budget_ratio - 1.0)
    return 0.70 * tool_agreement + 0.20 * verb_agreement + 0.10 * acting_rate


def analytic_comparison(chain_lengths=(1, 2, 3, 4), alpha=1.5):
    """Compare GAE attenuation/segment mass with direct balanced credit."""
    rows = []
    for k in chain_lengths:
        # Equal segments: CompactionRL lambda=1-1/(alpha*l), and l cancels in
        # the large-l limit: first-summary scale ~= exp(-(k-1)/alpha).
        gae_first = math.exp(-(k - 1) / alpha)
        w = event_weights(k)
        rows.append({
            "chain_length": k,
            "gae_first_vs_last_approx": gae_first,
            "segment_expanded_mass": float(k),
            "direct_total_mass": sum(w),
            "direct_first_vs_last": w[0] / w[-1],
        })
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps(analytic_comparison(), indent=2))
