#!/usr/bin/env python3
"""Plot the released 54-epoch vector-training history."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "figure_source" / "Fig_vector_training_54epoch_source_data.csv"
OUT = ROOT / "results" / "figures"


def main() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.0,
                         "axes.linewidth": 0.8, "axes.spines.top": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    if len(data) != 54 or set(data["record_epoch"]) != set(range(1, 55)):
        raise ValueError("Expected the frozen 54-epoch vector-training history")
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.85), constrained_layout=True)
    panels = [("train_loss", "Training loss", "#4f779e"), ("spearman", "Development Spearman", "#4f779e"),
              ("ndcg_at_10pct", "Development NDCG@10%", "#b6ccb9")]
    for ax, (column, ylabel, color) in zip(axes, panels):
        ax.plot(data["record_epoch"], data[column], marker="o", markersize=2.5, linewidth=1.15, color=color)
        ax.axvline(50.5, color="#7b8790", linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set(xlabel="Epoch", ylabel=ylabel, xlim=(1, 54))
        ax.grid(axis="y", color="#e7ecef", linewidth=0.5)
    axes[0].set_title("Frozen vector development history", fontsize=7.0)
    for ax in axes:
        ax.text(50.8, ax.get_ylim()[1], "low-LR branch", fontsize=6.0, color="#7b8790", va="top")
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"Fig_vector_training_dynamics.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
