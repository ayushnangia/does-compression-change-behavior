"""Statistical helpers for the experiments: bootstrap CIs and paired tests.

All our comparisons are within-example (every condition is measured on the
same decision points with shared seeds), so the correct tests are PAIRED:
bootstrap the per-example differences, or permute condition labels within
examples. These helpers assume aligned lists (one value per example).
"""

from __future__ import annotations

import random

try:  # citable reference implementations (Virtanen et al., Nature Methods 2020)
    import numpy as _np
    import scipy.stats as _st
    HAVE_SCIPY = True
except ImportError:  # pure-python fallbacks below keep the repo dependency-light
    HAVE_SCIPY = False


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """(lo, hi) percentile bootstrap CI for the mean of xs.
    scipy.stats.bootstrap when available; equivalent pure-python otherwise
    (agreement asserted in tests/run_tests.py)."""
    if len(xs) < 2:
        return float("nan"), float("nan")
    if HAVE_SCIPY:
        r = _st.bootstrap((_np.asarray(xs, dtype=float),), _np.mean,
                          n_resamples=n_boot, confidence_level=1 - alpha,
                          method="percentile",
                          rng=_np.random.default_rng(seed))
        return float(r.confidence_interval.low), float(r.confidence_interval.high)
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(mean([xs[rng.randrange(n)] for _ in range(n)])
                   for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def paired_permutation_p(a, b, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a - b) != 0 via sign-flipping the paired
    differences (a and b aligned per-example). scipy.stats.permutation_test
    (permutation_type='samples') when available; equivalent pure-python
    otherwise (agreement asserted in tests/run_tests.py)."""
    assert len(a) == len(b) and len(a) > 1
    diffs = [x - y for x, y in zip(a, b)]
    if HAVE_SCIPY:
        if all(abs(d) < 1e-12 for d in diffs):
            return 1.0  # degenerate: no variation to permute
        r = _st.permutation_test(
            (_np.asarray(diffs, dtype=float),), _np.mean,
            permutation_type="samples", alternative="two-sided",
            n_resamples=n_perm, rng=_np.random.default_rng(seed))
        return float(r.pvalue)
    rng = random.Random(seed)
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
