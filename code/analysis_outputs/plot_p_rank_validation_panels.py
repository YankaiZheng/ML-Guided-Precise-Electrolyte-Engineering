#!/usr/bin/env python3
"""Render manuscript P-rank counterparts of the D rank-validation panels.

Both panels consume frozen final predictions only.  The 4,000-molecule panel
uses the fixed evaluation subset; the external panel overlays the pre-specified
strict n=34 cohort within the common broad n=140 coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs"
OUT = ROOT / "analysis_outputs/paper_figures_ml_workflow"
SOURCE = OUT / "source_data"

EVALUATION4000 = RUNS / "domain65k_candidate100_final_locked/candidate100_clean_locked_dft_suitability_4194_random_remove194_seed20260715_kept4000_predictions.csv"
EXTERNAL = RUNS / "domain65k_candidate100_final_private_external179"
BROAD140 = EXTERNAL / "broad_structural_compatibility_n140_predictions.csv"
STRICT34 = EXTERNAL / "strict_domain_rule_matched_n34_predictions.csv"

TRUE_COL = "P"
PRED_COL = "P_pred__single_lgbm_raw"

COLORS = {
    "deep_blue": "#4f779e",
    "sky_blue": "#91b4d1",
    "green": "#b6ccb9",
    "mauve": "#c7b4be",
    "peach": "#e1aca6",
    "ink": "#263238",
    "mid_gray": "#7b8790",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.0,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.3,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.2,
        "ytick.major.size": 4.2,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def ranks_highest_first(values: pd.Series) -> np.ndarray:
    return rankdata(-np.asarray(values, dtype=float), method="average")


def ndcg10(y: np.ndarray, score: np.ndarray) -> float:
    k = max(1, int(np.ceil(len(y) * 0.10)))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    observed = y[np.argsort(score)[::-1][:k]]
    ideal = np.sort(y)[::-1][:k]
    return float(np.dot(observed, discount) / np.dot(ideal, discount))


def export(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_evaluation4000() -> dict:
    frame = pd.read_csv(EVALUATION4000)
    if len(frame) != 4000 or frame[[TRUE_COL, PRED_COL]].isna().any().any():
        raise AssertionError("Expected complete fixed 4,000-row P evaluation predictions")

    true_rank = ranks_highest_first(frame[TRUE_COL])
    pred_rank = ranks_highest_first(frame[PRED_COL])
    rho = float(spearmanr(frame[TRUE_COL], frame[PRED_COL]).statistic)
    ndcg = ndcg10(frame[TRUE_COL].to_numpy(float), frame[PRED_COL].to_numpy(float))

    fig, ax = plt.subplots(figsize=(4.95, 4.55), constrained_layout=True)
    bounds = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    cmap = ListedColormap(["#edf3f7", "#cad9e5", "#9fb9cd", "#7395b1", "#486d89"])
    density = ax.hexbin(
        true_rank, pred_rank, gridsize=72, extent=(1, 4000, 1, 4000), mincnt=2,
        norm=BoundaryNorm(bounds, cmap.N, clip=True), cmap=cmap, linewidths=0,
        edgecolors="none", rasterized=True,
    )
    diagonal = np.linspace(1, 4000, 400)
    ax.plot(diagonal, diagonal, color=COLORS["mid_gray"], linestyle=(0, (4, 3)), linewidth=1.05, label="Perfect ranking")
    slope, intercept = np.polyfit(true_rank, pred_rank, 1)
    ax.plot(diagonal, slope * diagonal + intercept, color=COLORS["deep_blue"], linewidth=1.65, label="Linear rank fit")
    ax.set(
        xlim=(1, 4000), ylim=(1, 4000), xticks=[1, 1000, 2000, 3000, 4000],
        yticks=[1, 1000, 2000, 3000, 4000], xlabel="True P rank (1 = highest)",
        ylabel="Predicted P rank (1 = highest)",
        title=f"P model: Rank prediction  |  Spearman ρ = {rho:.3f}",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.legend(loc="upper left", handlelength=2.1, labelspacing=0.32)
    cbar = fig.colorbar(density, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_title("Count", fontsize=6.3, pad=3)
    cbar.set_ticks([2, 3, 4, 5, 6], labels=["2", "3", "4", "5", "≥6"])
    export(fig, "Fig_P_rank_prediction_evaluation4000")

    source = frame[["locked_row_id", "canonical_smiles", TRUE_COL, PRED_COL]].copy()
    source["true_P_rank"] = true_rank
    source["predicted_P_rank"] = pred_rank
    source.to_csv(SOURCE / "Fig_P_rank_prediction_evaluation4000_source_data.csv", index=False)
    return {"n": len(frame), "spearman": rho, "ndcg_at_10pct": ndcg}


def plot_external140() -> dict:
    broad = pd.read_csv(BROAD140)
    strict = pd.read_csv(STRICT34)
    if len(broad) != 140 or len(strict) != 34:
        raise AssertionError("Expected frozen broad n=140 and strict n=34 cohorts")
    if broad[["canonical_smiles", TRUE_COL, PRED_COL]].isna().any().any() or strict[["canonical_smiles", TRUE_COL, PRED_COL]].isna().any().any():
        raise AssertionError("External P inputs contain missing values")
    broad_smiles = set(broad["canonical_smiles"])
    strict_smiles = set(strict["canonical_smiles"])
    if not strict_smiles.issubset(broad_smiles):
        raise AssertionError("Strict n=34 is not nested in broad n=140")

    rho_broad = float(spearmanr(broad[TRUE_COL], broad[PRED_COL]).statistic)
    rho_strict = float(spearmanr(strict[TRUE_COL], strict[PRED_COL]).statistic)
    ndcg_broad = ndcg10(broad[TRUE_COL].to_numpy(float), broad[PRED_COL].to_numpy(float))
    ndcg_strict = ndcg10(strict[TRUE_COL].to_numpy(float), strict[PRED_COL].to_numpy(float))

    broad = broad.copy()
    broad["true_P_rank"] = ranks_highest_first(broad[TRUE_COL])
    broad["predicted_P_rank"] = ranks_highest_first(broad[PRED_COL])
    broad["absolute_rank_error"] = np.abs(broad["true_P_rank"] - broad["predicted_P_rank"])
    broad["strict_65k_aligned"] = broad["canonical_smiles"].isin(strict_smiles)
    strict_common = broad.loc[broad["strict_65k_aligned"]].copy()

    # Error tiers match the D external panel: 10%, 20%, and 30% of n=140.
    masks = [
        broad["absolute_rank_error"] <= 14,
        (broad["absolute_rank_error"] > 14) & (broad["absolute_rank_error"] <= 28),
        (broad["absolute_rank_error"] > 28) & (broad["absolute_rank_error"] <= 42),
        broad["absolute_rank_error"] > 42,
    ]
    tier_colors = [COLORS["deep_blue"], COLORS["sky_blue"], COLORS["green"], COLORS["mauve"]]
    tier_names = ["Excellent (≤10%)", "Good (10–20%)", "Okay (20–30%)", "Large error (>30%)"]
    counts = [int(mask.sum()) for mask in masks]

    fig, ax = plt.subplots(figsize=(5.35, 4.95), constrained_layout=True)
    for mask, color in zip(masks, tier_colors):
        ax.scatter(
            broad.loc[mask, "true_P_rank"], broad.loc[mask, "predicted_P_rank"],
            s=28, c=color, edgecolors="white", linewidths=0.38, alpha=0.96, zorder=2,
        )
    ax.plot([1, 140], [1, 140], color=COLORS["mid_gray"], linewidth=1.05, linestyle=(0, (4, 3)), zorder=3)
    ax.scatter(
        strict_common["true_P_rank"], strict_common["predicted_P_rank"],
        s=53, facecolors="none", edgecolors=COLORS["peach"], linewidths=1.5, alpha=0.98, zorder=4,
    )
    ax.set(
        xlim=(0.5, 140.5), ylim=(0.5, 140.5), xticks=[1, 20, 60, 100, 140],
        yticks=[1, 20, 60, 100, 140], xlabel="True P rank (1 = highest)",
        ylabel="Predicted P rank (1 = highest)", title="External P: Rank prediction",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    error_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.4,
               markersize=5.7, label=f"{name}; n={count} ({100*count/140:.1f}%)")
        for name, color, count in zip(tier_names, tier_colors, counts)
    ]
    error_handles.append(Line2D([0], [0], color=COLORS["mid_gray"], linewidth=1.05, linestyle=(0, (4, 3)), label="Perfect ranking"))
    first = ax.legend(handles=error_handles, loc="upper left", handlelength=2.15, handletextpad=0.5, borderaxespad=0.5, labelspacing=0.30, fontsize=6.6)
    ax.add_artist(first)
    cohort_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=COLORS["mid_gray"], markeredgewidth=0.85,
               markersize=6.2, label=f"Broad structural-compatible set (n=140)\nSpearman = {rho_broad:.3f}; NDCG@10% = {ndcg_broad:.3f}"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=COLORS["peach"], markeredgewidth=1.45,
               markersize=6.8, label=f"Strict 65k-aligned subset (n=34)\nSpearman = {rho_strict:.3f}; NDCG@10% = {ndcg_strict:.3f}"),
    ]
    ax.legend(handles=cohort_handles, loc="lower right", handlelength=1.8, handletextpad=0.55, borderaxespad=0.55, labelspacing=0.70, fontsize=6.45)
    export(fig, "Fig_external_rule_stratified_P_rank_validation")

    broad[["canonical_smiles", "name", TRUE_COL, PRED_COL, "true_P_rank", "predicted_P_rank", "absolute_rank_error", "strict_65k_aligned"]].to_csv(
        SOURCE / "Fig_external_rule_stratified_P_rank_validation_source_data.csv", index=False
    )
    return {
        "broad": {"n": len(broad), "spearman": rho_broad, "ndcg_at_10pct": ndcg_broad},
        "strict": {"n": len(strict), "spearman": rho_strict, "ndcg_at_10pct": ndcg_strict, "nested_in_broad": True},
        "rank_error_counts_broad140": dict(zip(tier_names, counts)),
    }


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "P-model counterparts of the manuscript D rank-prediction and external rule-stratified validation panels.",
        "model": "Frozen final single-LightGBM P model.",
        "inputs": {"evaluation4000": str(EVALUATION4000), "broad140": str(BROAD140), "strict34": str(STRICT34)},
        "evaluation4000": plot_evaluation4000(),
        "external_rule_stratified": plot_external140(),
        "integrity": {"retrained": False, "test_labels_used_only_for_evaluation": True, "strict_group_nested_in_broad": True},
    }
    (OUT / "Fig_P_rank_validation_panels_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
