"""EXP 2 (analytic, no GPU) — Does treating compaction segments as samples
distort GRPO group statistics?

Claim being tested: "GRPO-style parallel rollouts ... can create over-reliance
on compaction-specific behavior and make credit assignment harder to
interpret." The paper argues the same qualitatively (their Sec. 'Ill-Suited
Group-Wise Methods'); this simulation puts numbers on it.

Setup: G rollouts per prompt share a group. Rollout g has reward r_g in {0,1}
and splits into K_g segments (all sharing r_g). If each segment is a group
sample, rollout g's reward enters the group mean/std K_g times AND receives
K_g gradient shares. We simulate two regimes:
  (a) K independent of outcome
  (b) K anti-correlated with success (harder tasks compact more and fail
      more — the realistic case)

    python exp2_grpo_weighting.py
"""

from __future__ import annotations

import random
import statistics


def group_advantages(rewards, k_segments, expand: bool):
    """Return per-rollout (advantage, gradient_mass) under GRPO normalization.
    expand=True treats each segment as a sample (reward repeated K_g times)."""
    pool = []
    for r, k in zip(rewards, k_segments):
        pool.extend([r] * (k if expand else 1))
    mu = statistics.mean(pool)
    sd = statistics.pstdev(pool) or 1.0
    out = []
    for r, k in zip(rewards, k_segments):
        adv = (r - mu) / sd
        mass = abs(adv) * (k if expand else 1)   # segments each carry adv
        out.append((adv, mass))
    return out


def simulate(correlated: bool, *, G=8, trials=20_000, seed=0):
    rng = random.Random(seed)
    d_adv_success = []          # advantage distortion for successful rollouts
    mass_by_k = {}              # gradient mass received, keyed by K_g
    for _ in range(trials):
        ks, rs = [], []
        for _ in range(G):
            k = rng.choice([1, 1, 2, 3, 4])
            if correlated:
                p_success = {1: 0.55, 2: 0.35, 3: 0.2, 4: 0.1}[k]
            else:
                p_success = 0.35
            ks.append(k)
            rs.append(1 if rng.random() < p_success else 0)
        if len(set(rs)) < 2:
            continue
        flat = group_advantages(rs, ks, expand=False)
        expd = group_advantages(rs, ks, expand=True)
        for (a0, _), (a1, m1), r, k in zip(flat, expd, rs, ks):
            if r == 1:
                d_adv_success.append(a1 - a0)
            mass_by_k.setdefault(k, []).append(m1)
    return d_adv_success, mass_by_k


def main():
    for name, corr in (("K independent of outcome", False),
                       ("K anti-correlated with success (realistic)", True)):
        d_adv, mass = simulate(corr)
        print(f"=== {name} ===")
        print(f"mean advantage shift for SUCCESSFUL rollouts when segments")
        print(f"are treated as samples: {statistics.mean(d_adv):+.3f}")
        print(f"{'K_g':>4s} {'mean gradient mass':>19s}")
        base = statistics.mean(mass[1])
        for k in sorted(mass):
            m = statistics.mean(mass[k])
            print(f"{k:4d} {m:19.3f}  ({m / base:.1f}x of K=1)")
        print()
    print("Reading: with segments-as-samples, a rollout that compacted 4x")
    print("carries several times the gradient mass of an uncompacted one for")
    print("the SAME outcome, and (in the realistic regime) the group mean is")
    print("dragged toward the outcomes of heavily-compacted rollouts, which")
    print("inflates advantages of successes. This is the quantitative case")
    print("for abandoning fixed-group normalization under compaction \u2014 it")
    print("supports both the claim and the paper's choice of PPO, while")
    print("showing WHY interpretability of credit suffers under GRPO here.")


if __name__ == "__main__":
    main()
