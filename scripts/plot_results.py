#!/usr/bin/env python
"""Bar charts of test AUROC, accuracy and F1 (relation-balanced), mean and sd over seeds.

Bars: the four KGE baselines, PNKG with random coordinates, the per-seed average of the single views, and
the multi-view PNKG. A star marks multi-view above the single-view average with a paired t-test p < 0.05.

    python scripts/plot_results.py wordnet --summary results/wordnet/summary.csv --out figures/wordnet.pdf
    python scripts/plot_results.py renki --results-dir results/renki --out figures/renki.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ttest_rel  # noqa: E402

COLORS = ["#74a9cf", "#a1d99b", "#9e9ac8", "#fc8d59", "#bdbdbd", "#2171b5", "#e6550d"]
LABELS = ["TransE", "DistMult", "ComplEx", "RotatE", "PNKG: rand"]


def plot(kge, rand, single, multi, n_views, out, zoom):
    """kge: four per-seed frames (TransE, DistMult, ComplEx, RotatE); columns AUROC, Accuracy, F1."""
    bars = list(zip(LABELS, [*kge, rand])) + [(f"PNKG: single-view average ({n_views} views)", single),
                                              ("PNKG: multi-view", multi)]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), sharey=not zoom)
    for ax, col in zip(axes, ["AUROC", "Accuracy", "F1"]):
        means = [df[col].mean() for _, df in bars]
        lo, hi = 0.0, 1.12
        if zoom:
            lo = np.floor((min(means) - 0.03) * 20) / 20
            hi = max(df[col].mean() + df[col].std() for _, df in bars) + 0.1 * (max(means) - lo)
        for i, ((label, df), color) in enumerate(zip(bars, COLORS)):
            m, s = df[col].mean(), df[col].std()
            ax.bar(i, m, yerr=s, color=color, label=label, width=0.85, capsize=3,
                   error_kw=dict(ecolor="#444444", elinewidth=1.2))
            ax.text(i, m + s + (0.002 * (hi - lo) / 0.1 if zoom else 0.01), f"{m:.3f}", ha="center", va="bottom", fontsize=10)
        seeds = multi.index.intersection(single.index)
        a, b = multi.loc[seeds, col], single.loc[seeds, col]
        if a.mean() > b.mean() and ttest_rel(a, b).pvalue < 0.05:
            top = multi[col].mean() + multi[col].std()
            ax.text(len(bars) - 1, top + 0.045 * (hi - lo if zoom else 1), "*", ha="center", va="bottom",
                    color=COLORS[-1], fontsize=18)
        ax.set_title(col, fontsize=16, fontweight="bold")
        ax.set_xticks([])
        ax.set_ylim(lo, hi)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Score", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(bars), fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(pd.DataFrame({label: df.agg(["mean", "std"]).T.stack() for label, df in bars}).T.round(4).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="dataset", required=True)
    w = sub.add_parser("wordnet")
    w.add_argument("--summary", type=Path, default=Path("results/wordnet/summary.csv"))
    w.add_argument("--single", nargs="+", default=["bert", "roberta", "electra", "deberta"])
    w.add_argument("--multiview", default="selected")
    w.add_argument("--out", type=Path, default=Path("figures/wordnet.pdf"))
    r = sub.add_parser("renki")
    r.add_argument("--results-dir", type=Path, default=Path("results/renki"))
    r.add_argument("--single", nargs="+",
                   default=[f"pca128_{v}" for v in ["bert", "roberta", "nerbert", "mentalbert", "scibert", "biobert", "clinicalbert"]])
    r.add_argument("--multiview", default="mv7_d128")
    r.add_argument("--out", type=Path, default=Path("figures/renki.pdf"))
    args = p.parse_args()

    if args.dataset == "wordnet":
        df = pd.read_csv(args.summary).rename(columns={"test_auc": "AUROC", "test_acc": "Accuracy", "test_f1": "F1"})
        read = lambda m: df[df["method"] == m].set_index("seed")[["AUROC", "Accuracy", "F1"]]
        zoom = False
    else:
        cols = {"auroc_rb": "AUROC", "acc_rb": "Accuracy", "f1_rb": "F1"}
        read = lambda m: pd.read_csv(args.results_dir / f"{m}.csv", index_col="seed")[list(cols)].rename(columns=cols)
        zoom = True
    single = (sum(read(m) for m in args.single) / len(args.single)).dropna()
    kge = [read(m) for m in ["transe", "distmult", "complex", "rotate"]]
    plot(kge, read("random"), single, read(args.multiview), len(args.single), args.out, zoom)


if __name__ == "__main__":
    main()
