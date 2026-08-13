#!/usr/bin/env python3
"""Plot the public Test-4000 D-ranking algorithm comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "figure_source" / "fig2b_algorithm_evaluation4000.csv"
OUT = ROOT / "results" / "figures"

COLORS = {
    "deep_blue": "#4f779e",
    "sky_blue": "#91b4d1",
    "green": "#b6ccb9",
    "mauve": "#c7b4be",
    "ink": "#263238",
    "mid_gray": "#7b8790",
    "light_gray": "#dce3e7",
    "near_white": "#f4f7f9",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.titlesize": 8.3,
        "axes.labelsize": 7.6,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def model_color(model: str) -> str:
    if model == "Final fusion":
        return COLORS["deep_blue"]
    if model == "Equivariant vector":
        return COLORS["green"]
    if "LTR" in model:
        return COLORS["light_gray"]
    if "MLP" in model or "ResNet" in model:
        return COLORS["mauve"]
    if "kNN" in model:
        return "#b8c9d6"
    if "xTB direct" in model:
        return COLORS["mid_gray"]
    return COLORS["sky_blue"]


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    order = [
        "Final fusion", "XGBoost", "LightGBM", "CatBoost", "RealMLP",
        "Equivariant vector", "ResNet-RTDL", "CatBoost-YetiRank",
        "LightGBM-LTR", "XGBoost-LTR", "Morgan-Tanimoto kNN", "GFN2-xTB direct",
    ]
    data["order"] = data["model"].map({name: idx for idx, name in enumerate(order)})
    if data["order"].isna().any() or len(data) != len(order):
        raise RuntimeError("The published comparison table is incomplete or contains an unknown model")
    data = data.sort_values("order").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True, gridspec_kw={"wspace": 0.10})
    y = np.arange(len(data))[::-1]
    panels = [
        ("spearman", "Spearman correlation", (0.50, 0.85), [0.50, 0.60, 0.70, 0.80]),
        ("ndcg_at_10pct", "NDCG@10%", (0.79, 0.905), [0.80, 0.84, 0.88]),
    ]
    for panel_idx, (ax, (column, label, limits, ticks)) in enumerate(zip(axes, panels)):
        values = data[column].to_numpy(float)
        for idx, (value, model) in enumerate(zip(values, data["model"])):
            color = model_color(model)
            base = limits[0]
            ax.plot([base, value], [y[idx], y[idx]], color="#e2e7ea", linewidth=1.0, zorder=1)
            ax.scatter(
                value, y[idx], s=54 if model == "Final fusion" else 38,
                color=color, edgecolor="white", linewidth=0.65, zorder=3,
            )
            offset = (limits[1] - limits[0]) * 0.018
            ax.text(
                value + offset, y[idx], f"{value:.3f}", va="center", ha="left",
                fontsize=6.1, color=COLORS["ink"],
                fontweight="bold" if model == "Final fusion" else "normal",
            )
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_xlabel(label)
        ax.grid(False)
        ax.grid(axis="x", color="#e7ecef", linewidth=0.55)
        ax.tick_params(axis="y", length=0)
        ax.set_title(
            "Global rank agreement" if panel_idx == 0 else "Top-decile screening quality",
            loc="left", pad=7, fontweight="bold",
        )
        ax.text(-0.10 if panel_idx == 0 else -0.04, 1.055, "ab"[panel_idx], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
    axes[0].set_yticks(y, data["model"])
    axes[1].tick_params(axis="y", labelleft=False)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.2,
                   markerfacecolor=COLORS["sky_blue"], markeredgecolor="white", label="Tree boosting"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.2,
                   markerfacecolor=COLORS["mauve"], markeredgecolor="white", label="Deep tabular"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.2,
                   markerfacecolor=COLORS["green"], markeredgecolor="white", label="Physics-guided"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.2,
                   markerfacecolor="#b8c9d6", markeredgecolor="white", label="Similarity baseline"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.2,
                   markerfacecolor=COLORS["mid_gray"], markeredgecolor="white", label="Direct xTB"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.2,
                   markerfacecolor=COLORS["light_gray"], markeredgecolor="white", label="LTR"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.2,
                   markerfacecolor=COLORS["deep_blue"], markeredgecolor="white", label="Final system"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.01), ncol=4,
               columnspacing=1.10, handletextpad=0.40)
    fig.text(0.54, 0.105, "Frozen Test-4000 cohort",
             ha="center", va="bottom", fontsize=6.4, color=COLORS["mid_gray"])
    fig.subplots_adjust(left=0.24, right=0.975, top=0.88, bottom=0.28)

    stem = OUT / "Fig2b_algorithm_comparison_evaluation4000"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "surface_class": "paper_main",
        "core_claim": "The frozen final fusion integrates complementary global-rank and top-decile signals.",
        "source_data": str(INPUT),
        "generating_script": "analysis_outputs/plot_domain65k_d_algorithm_comparison_4000.py",
        "exports": [f"{stem.name}.png", f"{stem.name}.pdf", f"{stem.name}.svg"],
        "cohort": "Frozen Test-4000 evaluation cohort.",
    }
    (OUT / f"{stem.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
