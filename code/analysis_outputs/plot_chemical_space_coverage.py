#!/usr/bin/env python3
"""Render the released Morgan-fingerprint chemical-space coverage panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "figure_source"
OUT = ROOT / "results" / "figures"

COLORS = {"development": "#4f779e", "evaluation": "#b6ccb9", "candidates": "#e1aca6", "ink": "#263238"}


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    OUT.mkdir(parents=True, exist_ok=True)
    coordinates = pd.read_csv(SOURCE / "fig2_chemical_space_projection.csv")
    similarity = pd.read_csv(SOURCE / "fig2_similarity_coverage_curve.csv")
    if coordinates.groupby("group").size().to_dict() != {"Candidates": 78, "Development": 60600, "Evaluation": 4000}:
        raise ValueError("Unexpected public chemical-space cohort sizes")
    if not {"Evaluation", "Candidates"}.issubset(set(similarity["group"])):
        raise ValueError("Similarity coverage table is incomplete")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": [1.3, 1]}, constrained_layout=True)
    ax = axes[0]
    development = coordinates.loc[coordinates["group"].eq("Development")]
    evaluation = coordinates.loc[coordinates["group"].eq("Evaluation")]
    candidates = coordinates.loc[coordinates["group"].eq("Candidates")]
    density = ax.hexbin(development["latent_1"], development["latent_2"], gridsize=95, bins="log", mincnt=1,
                        cmap="Blues", linewidths=0, rasterized=True)
    ax.scatter(evaluation["latent_1"], evaluation["latent_2"], s=3.0, c=COLORS["evaluation"], alpha=0.35,
               linewidths=0, label="Test-4000")
    ax.scatter(candidates["latent_1"], candidates["latent_2"], s=20, c=COLORS["candidates"], edgecolors="white",
               linewidths=0.35, label="78 candidates", zorder=3)
    ax.set(xlabel="Morgan fingerprint latent axis 1", ylabel="Morgan fingerprint latent axis 2", title="Chemical-space atlas")
    ax.legend(frameon=False, loc="upper right", fontsize=6.8)
    cbar = fig.colorbar(density, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_title("Development\ndensity", fontsize=6.2, pad=3)

    ax = axes[1]
    for group, color, label in (("Evaluation", COLORS["development"], "Test-4000"), ("Candidates", COLORS["candidates"], "78 candidates")):
        subset = similarity.loc[similarity["group"].eq(group)].sort_values("threshold")
        ax.plot(subset["threshold"], subset["coverage"], color=color, linewidth=1.8, label=label)
    ax.axvline(0.60, color="#7b8790", linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set(xlabel="Nearest-development Tanimoto threshold", ylabel="Coverage fraction", ylim=(0, 1.03),
           title="Nearest-neighbour coverage")
    ax.legend(frameon=False, loc="lower left", fontsize=6.8)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"Fig_chemical_space_coverage.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
