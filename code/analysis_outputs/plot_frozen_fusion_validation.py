#!/usr/bin/env python3
"""Plot the prespecified development-validation member comparison and weights."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "figure_source"
OUT = ROOT / "results" / "figures"


def main() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.0,
                         "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    OUT.mkdir(parents=True, exist_ok=True)
    members = pd.read_csv(SOURCE / "fig4_fold0_member_metrics.csv")
    weights = pd.read_csv(SOURCE / "fig4_fusion_weights.csv")
    if len(members) != 9 or len(weights) != 8 or not np.isclose(weights["weight"].sum(), 1.0, atol=1e-12):
        raise ValueError("Published fusion validation tables are inconsistent")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), constrained_layout=True)
    x = np.arange(len(members))
    width = 0.38
    axes[0].bar(x - width / 2, members["spearman"], width, color="#4f779e", label="Spearman")
    axes[0].bar(x + width / 2, members["ndcg_at_10pct"], width, color="#b6ccb9", label="NDCG@10%")
    axes[0].set(xticks=x, xticklabels=members["label"], ylim=(0.70, 0.95), ylabel="Development-validation score",
                title="Frozen member comparison")
    axes[0].tick_params(axis="x", rotation=45, labelsize=6.2)
    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=6.7)
    active = weights.sort_values("weight", ascending=True)
    axes[1].barh(active["label"], active["weight"], color=["#e1aca6" if value == 0 else "#4f779e" for value in active["weight"]])
    axes[1].set(xlabel="Frozen fusion weight", title="Eight-member rank fusion")
    axes[1].axvline(0, color="#263238", linewidth=0.6)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"Fig_frozen_fusion_validation.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
