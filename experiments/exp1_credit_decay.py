"""EXP 1 (analytic, no GPU) — How much weaker is the learning signal for
earlier summaries under CompactionRL's cross-trajectory GAE?

Claim being tested: "as the compaction chain gets deeper, earlier summaries
receive a much weaker learning signal, while the final few compactions are
much more directly responsible for the answer."

The paper (arXiv:2607.05378) corrects segment-local GAE with
    A_hat[s,i] = (gamma*lambda)^{N>s} * A_loc[s,i]
where N>s is the number of optimized tokens generated AFTER segment s, and
uses length-adaptive lambda = 1 - 1/(alpha*l) with alpha = 1.5 (their Sec. 4).
Summary tokens sit at the END of their segment, so their distance to the
terminal reward is ~N>s. This script computes the attenuation exactly.

    python exp1_credit_decay.py
"""

from __future__ import annotations


def attenuation(n_later_tokens: int, seg_len: int, alpha: float = 1.5,
                gamma: float = 1.0) -> float:
    """(gamma*lambda)^{N>s} with the paper's length-adaptive lambda."""
    lam = 1.0 - 1.0 / (alpha * seg_len)
    return (gamma * lam) ** n_later_tokens


def main():
    print("Cross-trajectory GAE attenuation of the SUMMARY tokens in segment s")
    print("(paper hyperparams: lambda = 1 - 1/(1.5*l), gamma = 1)\n")

    for seg_len in (5_000, 10_000, 20_000):
        print(f"--- equal segment length l = {seg_len} tokens ---")
        print(f"{'chain K':>8s} {'segment s':>10s} {'N>s':>9s} "
              f"{'signal scale':>13s} {'vs last seg':>12s}")
        for K in (2, 3, 4):
            scales = []
            for s in range(1, K + 1):
                n_later = (K - s) * seg_len
                scales.append(attenuation(n_later, seg_len))
            for s, a in enumerate(scales, 1):
                rel = a / scales[-1]
                print(f"{K:8d} {s:10d} {(K - s) * seg_len:9d} "
                      f"{a:13.4f} {rel:11.1%}")
            print()

    print("Reading: in a 4-segment chain the FIRST summary's gradient signal")
    print("is scaled to ~14% of the last segment's — the claim is not an")
    print("artifact, it is the designed behavior of the discount. The open")
    print("question is whether discounted credit is the right prior for")
    print("summaries at all: a summary gates ALL downstream information, so")
    print("its causal weight arguably does not decay like an action's.")
    print("(The paper explicitly rejects auxiliary summary rewards, so this")
    print("attenuation is uncompensated.)")


if __name__ == "__main__":
    main()
