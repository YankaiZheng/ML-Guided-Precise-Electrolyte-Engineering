#!/usr/bin/env python3
"""Render the public Test-4000 D-rank agreement panel from released source data."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "figure_source" / "fig5_rank_density_source.csv"
OUT = ROOT / "results" / "figures"


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0, "axes.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    if len(data) != 4000 or data[["D", "fusion_score", "true_rank", "predicted_rank"]].isna().any().any():
        raise ValueError("Expected 4,000 complete public rank-density records")
    rho = float(spearmanr(data["D"], data["fusion_score"]).statistic)
    fig, ax = plt.subplots(figsize=(5.0, 4.55), constrained_layout=True)
    cmap = ListedColormap(["#edf3f7", "#cad9e5", "#9fb9cd", "#7395b1", "#486d89"])
    density = ax.hexbin(
        data["true_rank"], data["predicted_rank"], gridsize=72, extent=(1, 4000, 1, 4000), mincnt=2,
        norm=BoundaryNorm([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N, clip=True), cmap=cmap, linewidths=0,
        rasterized=True,
    )
    diagonal = np.linspace(1, 4000, 400)
    ax.plot(diagonal, diagonal, color="#7b8790", linestyle=(0, (4, 3)), linewidth=1.05, label="Perfect ranking")
    slope, intercept = np.polyfit(data["true_rank"], data["predicted_rank"], 1)
    ax.plot(diagonal, slope * diagonal + intercept, color="#4f779e", linewidth=1.65, label="Linear rank fit")
    ax.set(xlim=(1, 4000), ylim=(1, 4000), xticks=[1, 1000, 2000, 3000, 4000], yticks=[1, 1000, 2000, 3000, 4000],
           xlabel="True D rank (1 = highest)", ylabel="Predicted D rank (1 = highest)",
           title=f"Ranking consistency  |  Spearman ρ = {rho:.3f}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False)
    cbar = fig.colorbar(density, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_title("Count", fontsize=6.3, pad=3)
    cbar.set_ticks([2, 3, 4, 5, 6], labels=["2", "3", "4", "5", "≥6"])
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"Fig_D_rank_prediction_evaluation4000.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
