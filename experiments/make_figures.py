"""Generate all showcase figures from experiments/results/*.json and
results/result_*.json. Plots whatever exists, skips what doesn't.

    python make_figures.py            # writes PNGs to experiments/figures/
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

FLOOR_STYLE = dict(color="gray", linestyle="--", linewidth=1)


def latest(pattern: str, root: Path):
    files = sorted(root.glob(pattern))
    return json.loads(files[-1].read_text()) if files else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote figures/{name}.png")


def fig_credit_decay():
    """exp1 is analytic — recompute here so the figure is self-contained."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    K = 4
    for alpha, marker in ((1.5, "o"),):
        xs = list(range(1, K + 1))
        ys = [math.exp(-(K - s) / alpha) for s in xs]
        ax.plot(xs, ys, marker=marker, label=f"paper's \u03b1={alpha}")
    ax.set_xticks(range(1, K + 1))
    ax.set_xlabel("segment position s (chain of 4 compactions)")
    ax.set_ylabel("learning-signal scale of segment's summary")
    ax.set_title("CompactionRL cross-trajectory GAE:\nearly summaries train at a fraction of late ones")
    ax.axhline(1.0, **FLOOR_STYLE)
    ax.legend()
    save(fig, "fig1_credit_decay")


def fig_target_stability():
    runs = sorted((HERE / "results").glob("exp3_target_stability_*.json"))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for f in runs:
        d = json.loads(f.read_text())
        Ls = sorted(int(k) for k in d["drift_vs_max"])
        ax.plot(Ls, [d["drift_vs_max"][str(L)] for L in Ls], marker="o",
                label=f"{d['model'].split('/')[-1]} (floor {d['floor']:.2f})")
        ax.axhline(d["floor"], **FLOOR_STYLE)
    ax.set_xlabel("truncated context length (tokens)")
    ax.set_ylabel("behavior drift vs full 4k context")
    ax.set_title("Is the full-context target stable?\n(drift near floor = trustworthy SSL target)")
    ax.legend()
    save(fig, "fig2_target_stability")


def fig_block_ablation():
    d = latest("results/exp4_block_ablation_*.json", HERE)
    if not d:
        return
    names = list(d["variants"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, title in ((axes[0], "change", "behavior change (lower = better)"),
                           (axes[1], "none", "halt rate (fraction no-action)")):
        vals = [d["variants"][n][key] for n in names]
        ax.bar(range(len(names)), vals,
                      color=["tab:gray", "tab:red", "tab:orange", "tab:green"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("drop_", "\u2212") for n in names], rotation=15)
        ax.set_title(title)
        if key == "change":
            ax.axhline(d["floor"], **FLOOR_STYLE)
            ax.text(0.02, d["floor"] + 0.01, "sampling floor", fontsize=7, color="gray")
    fig.suptitle(f"Equal 50% token budget \u2014 which blocks carry behavior? ({d['model'].split('/')[-1]})")
    save(fig, "fig3_block_ablation")


def fig_format():
    d = latest("results/exp5_format_vs_content_*.json", HERE)
    if not d:
        return
    names = list(d["formats"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [d["formats"][n]["acting"] for n in names],
           width=0.4, label="acting rate", color="tab:blue")
    ax.bar([i + 0.2 for i in x], [d["formats"][n]["change"] for n in names],
           width=0.4, label="behavior change", color="tab:red")
    ax.axhline(d["floor"], **FLOOR_STYLE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15)
    ax.set_title("Identical summary content, different interface format\n(format alone moves behavior)")
    ax.legend()
    save(fig, "fig4_format_vs_content")


def fig_rate_distortion():
    d = latest("results/exp6_rate_distortion_*.json", HERE)
    if not d:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    comps = sorted({k.split("@")[0] for k in d["curves"]})
    for c in comps:
        rs = sorted(float(k.split("@")[1]) for k in d["curves"] if k.startswith(c + "@"))
        ys = [d["curves"][f"{c}@{r:g}"]["change"] for r in rs]
        ax.plot(rs, ys, marker="o", label=c)
    ax.axhline(d["floor"], **FLOOR_STYLE)
    ax.text(0.02, d["floor"] + 0.01, "sampling floor", fontsize=7, color="gray")
    ax.set_xlabel("rate R (fraction of history kept)")
    ax.set_ylabel("distortion D (behavior change)")
    ax.set_title(f"The rate\u2013distortion curve ({d['model'].split('/')[-1]})")
    ax.legend()
    save(fig, "fig5_rate_distortion")


def fig_chain():
    d = latest("results/exp7_compaction_chain_*.json", HERE)
    if not d:
        return
    ks = sorted(int(k) for k in d["chain"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(ks, [d["chain"][str(k)]["change"] for k in ks], marker="o", color="tab:red")
    axes[0].axhline(d["floor"], **FLOOR_STYLE)
    axes[0].set_xlabel("compaction chain depth k")
    axes[0].set_ylabel("behavior change vs full context")
    axes[0].set_title("does distortion compound?")
    axes[1].plot(ks, [d["chain"][str(k)]["entropy"] for k in ks], marker="s", color="tab:purple",
                 label="compressed")
    axes[1].axhline(d["full_entropy"], color="tab:blue", linestyle=":", label="full context")
    axes[1].set_xlabel("compaction chain depth k")
    axes[1].set_ylabel("action entropy (bits)")
    axes[1].set_title("is uncertainty structure preserved?")
    axes[1].legend()
    fig.suptitle(f"Repeated compaction ({d['model'].split('/')[-1]}, budget {d['summary_budget']} tok)")
    save(fig, "fig6_compaction_chain")


def fig_main_table():
    d = latest("result_*.json", REPO / "results")
    if not d:
        return
    comps = d["compressors"]
    names = [n for n in ("full", "keep_recent", "summary", "paraphrase",
                         "pointer", "hallucinator") if n in comps]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [comps[n]["acting_rate"] for n in names],
           width=0.4, label="acting rate", color="tab:blue")
    ax.bar([i + 0.2 for i in x], [comps[n]["action_change"] for n in names],
           width=0.4, label="behavior change", color="tab:red")
    ax.axhline(d["floor"], **FLOOR_STYLE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15)
    mode = "recovery scaffold" if d.get("scaffold") else "plain next-action"
    ax.set_title(f"All compressors ({d['model'].split('/')[-1]}, {mode})\n"
                 "pointer (recoverable) vs hallucinator (unrecoverable) is the metric's honesty test")
    ax.legend()
    save(fig, "fig7_all_compressors")


def fig_grounded():
    d = latest("results/exp8_grounded_agreement_*.json", HERE)
    if not d:
        return
    names = list(d["compressors"])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [d["compressors"][n]["tool"] for n in names],
           width=0.4, label="agreement with REAL action", color="tab:green")
    ax.bar([i + 0.2 for i in x], [d["compressors"][n]["change"] for n in names],
           width=0.4, label="behavior change D", color="tab:red")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15)
    ax.set_title("D is grounded: high change = low agreement with the\n"
                 "action the original agent actually took")
    ax.legend()
    save(fig, "fig8_grounded_agreement")


def fig_policies():
    d = latest("results/exp9_summary_policies_*.json", HERE)
    if not d:
        return
    names = sorted(d["policies"], key=lambda p: d["policies"][p]["agree"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [d["policies"][n]["acting"] for n in names],
           width=0.4, label="acting rate", color="tab:blue")
    ax.bar([i + 0.2 for i in x], [d["policies"][n]["agree"] for n in names],
           width=0.4, label="agreement w/ real action", color="tab:green")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_title("Summary policies: the prompt derived from our block\n"
                 "measurements (block_aware) beats deployed-style prompts")
    ax.legend()
    save(fig, "fig9_summary_policies")


def fig_propagation():
    d = latest("results/exp10_propagation_*.json", HERE)
    if not d:
        return
    ks = sorted(int(k) for k in d["steps"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(ks, [d["steps"][str(k)]["change"] for k in ks], marker="o",
            color="tab:red", label="behavior change vs full")
    ax.plot(ks, [d["steps"][str(k)]["floor"] for k in ks], linestyle="--",
            color="gray", label="sampling floor")
    ax.plot(ks, [d["steps"][str(k)]["acting"] for k in ks], marker="s",
            color="tab:blue", label="acting rate")
    ax.set_xticks(ks)
    ax.set_xlabel("real trajectory steps after ONE compaction")
    ax.set_title("Compaction damage HEALS under the real continuation\n"
                 "(teacher-forced — upper bound on healing)")
    ax.legend(fontsize=8)
    save(fig, "fig10_propagation")


def fig_selection():
    d = latest("results/exp11_best_of_n_*.json", HERE)
    if not d:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    metrics = [("immediate_d", "immediate D ↓"), ("grounded_agree", "real-action agreement ↑"),
               ("downstream_d", "downstream D ↓")]
    x = range(len(metrics))
    ax.bar([i - 0.2 for i in x], [d[m]["best"] for m, _ in metrics],
           width=0.4, label="select-by-D (best of 8)", color="tab:green")
    ax.bar([i + 0.2 for i in x], [d[m]["random"] for m, _ in metrics],
           width=0.4, label="random pick", color="tab:gray")
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=8)
    ax.set_title("Behavioral distortion works as a SELECTION signal\n"
                 "(fresh-scored; no training)")
    ax.legend()
    save(fig, "fig11_selection")


def fig_manifest():
    d = latest("results/exp13_manifest_*.json", HERE)
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, mode, title in ((axes[0], "plain", "plain next-action"),
                            (axes[1], "scaffold", "recovery scaffold")):
        rows = d.get(mode) or {}
        if not rows:
            continue
        names = list(rows)
        ax.bar(range(len(names)), [rows[n]["acting"] for n in names],
               color=["tab:gray", "tab:orange", "tab:purple"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.set_ylabel("acting rate")
        ax.set_title(title)
    fig.suptitle("Deletion-manifest idea: mostly null at this scale (honest result)")
    save(fig, "fig13_manifest")


if __name__ == "__main__":
    for fn in (fig_credit_decay, fig_target_stability, fig_block_ablation,
               fig_format, fig_rate_distortion, fig_chain, fig_main_table,
               fig_grounded, fig_policies, fig_propagation, fig_selection,
               fig_manifest):
        try:
            fn()
        except Exception as e:  # keep going: plot everything that exists
            print(f"{fn.__name__}: skipped ({e})")
