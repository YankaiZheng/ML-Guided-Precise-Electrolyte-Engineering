#!/usr/bin/env python3
"""Render the public target-free multiscale feature-correlation heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "figure_source" / "feature_relationship_spearman_matrix.csv"
OUT = ROOT / "results" / "figures"

COLORS = {
    "deep_blue": "#4f779e",
    "peach": "#e1aca6",
    "ink": "#263238",
    "near_white": "#f7f9fa",
}

SHORT = ["MW", "O", "F", "Rings", "RotB", "TPSA", "logP", "Δq", "Rg", "μ", "α", "σμ 3C", "AR μq"]


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT, index_col=0)
    matrix = frame.to_numpy(float)
    if matrix.shape != (13, 13) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Expected a symmetric 13 by 13 published correlation matrix")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12):
        raise ValueError("Correlation diagonal must equal one")

    cmap = LinearSegmentedColormap.from_list(
        "correlation", [COLORS["deep_blue"], "#9eb8ca", COLORS["near_white"], COLORS["peach"], "#c87670"]
    )
    fig = plt.figure(figsize=(4.55, 4.28))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.045], hspace=0.34)
    ax = fig.add_subplot(grid[0])
    cax = fig.add_subplot(grid[1])
    image = ax.imshow(matrix, cmap=cmap, norm=Normalize(vmin=-1, vmax=1), interpolation="nearest", aspect="equal")
    ax.set_xticks(np.arange(13), SHORT, rotation=42, ha="right", rotation_mode="anchor", fontsize=5.7)
    ax.set_yticks(np.arange(13), frame.index, fontsize=6.1)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 13, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.58)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(13):
        for column in range(13):
            value = matrix[row, column]
            if row != column and abs(value) >= 0.48:
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=4.35, color=COLORS["ink"])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    colorbar.ax.tick_params(labelsize=5.8, length=2)
    colorbar.set_label("Target-free Spearman correlation across 60,641 development records", fontsize=6.0, labelpad=2)
    fig.suptitle("Multiscale feature complementarity", x=0.50, y=0.985, fontsize=10.2, fontweight="bold", color=COLORS["deep_blue"])
    fig.subplots_adjust(left=0.27, right=0.98, top=0.79, bottom=0.15)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"Fig_feature_complementarity_heatmap.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
