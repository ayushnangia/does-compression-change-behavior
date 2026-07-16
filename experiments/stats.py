"""Statistical helpers for the experiments: bootstrap CIs and paired tests.

All our comparisons are within-example (every condition is measured on the
same decision points with shared seeds), so the correct tests are PAIRED:
bootstrap the per-example differences, or permute condition labels within
examples. These helpers assume aligned lists (one value per example).
"""

from __future__ import annotations

import random


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """(lo, hi) percentile bootstrap CI for the mean of xs."""
    if len(xs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(mean([xs[rng.randrange(n)] for _ in range(n)])
                   for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def paired_permutation_p(a, b, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a - b) != 0 via sign-flipping the paired
    differences. a and b must be aligned per-example lists."""
    assert len(a) == len(b) and len(a) > 1
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    obs = abs(mean(diffs))
    hits = 0
    for _ in range(n_perm):
        s = mean([d if rng.random() < 0.5 else -d for d in diffs])
        if abs(s) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def fmt_ci(xs) -> str:
    lo, hi = bootstrap_ci(xs)
    return f"{mean(xs):.2f} [{lo:.2f},{hi:.2f}]"
