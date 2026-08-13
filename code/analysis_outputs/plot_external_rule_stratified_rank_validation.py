#!/usr/bin/env python3
"""Plot the nested broad/strict external D-rank validation panel.

This script consumes only frozen external predictions.  The broad n=140 cohort
provides a single common coordinate system.  The strict n=34 cohort is a
pre-specified subset and is overlaid in that same coordinate system; its
reported metric is nevertheless recomputed under the frozen rank fusion within
the n=34 cohort, matching the accompanying external-validation table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures"
SOURCE = OUT / "source_data"

BROAD_CSV = ROOT / "data" / "external" / "external_broad_n140.csv"
STRICT_CSV = ROOT / "data" / "external" / "external_strict_n34.csv"
WEIGHTS_JSON = ROOT / "code" / "frozen_configs" / "final_d_fusion_weights.json"

STEM = "Fig_external_rule_stratified_rank_validation"
SCORE_COL = "D_score__recomputed_subset"

COLORS = {
    "deep_blue": "#4f779e",
    "sky_blue": "#91b4d1",
    "peach": "#e1aca6",
    "ink": "#263238",
    "mid_gray": "#7b8790",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
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
        }
    )


def descending_rank_percentile(values: pd.Series) -> np.ndarray:
    """Map high D or score to 100 and low values to 0, preserving ties."""
    values = np.asarray(values, dtype=float)
    rank = rankdata(-values, method="average")
    return 100.0 * (len(values) - rank) / (len(values) - 1)


def descending_rank(values: pd.Series) -> np.ndarray:
    """Return ranks where 1 denotes the largest D or predicted score."""
    return rankdata(-np.asarray(values, dtype=float), method="average")


def ndcg_at_10pct(y_true: np.ndarray, score: np.ndarray) -> float:
    k = max(1, int(np.ceil(len(y_true) * 0.10)))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    predicted = np.argsort(-score, kind="mergesort")[:k]
    ideal = np.argsort(-y_true, kind="mergesort")[:k]
    # Use raw physical-property values as gains, matching the frozen report.
    gains = np.asarray(y_true, dtype=float)
    return float(np.dot(gains[predicted], discount) / np.dot(gains[ideal], discount))


def require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=1e-12, rtol=0.0):
        raise AssertionError(f"{label}: expected {expected:.12f}, observed {actual:.12f}")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    broad = pd.read_csv(BROAD_CSV)
    strict = pd.read_csv(STRICT_CSV)
    weights_payload = json.loads(WEIGHTS_JSON.read_text(encoding="utf-8"))

    for label, frame, expected_n in (("broad", broad, 140), ("strict", strict, 34)):
        if len(frame) != expected_n or frame["canonical_smiles"].nunique() != expected_n:
            raise AssertionError(f"{label} cohort must contain {expected_n} unique canonical SMILES")
        if frame["canonical_smiles"].duplicated().any():
            raise AssertionError(f"{label} cohort has duplicate canonical SMILES")
        if frame[["D", SCORE_COL]].isna().any().any():
            raise AssertionError(f"{label} cohort has missing D or fusion score")

    broad_smiles = set(broad["canonical_smiles"])
    strict_smiles = set(strict["canonical_smiles"])
    if not strict_smiles.issubset(broad_smiles):
        raise AssertionError("Strict n=34 must be a subset of broad n=140")

    # These metrics match the previously frozen external-validation report.
    rho_broad = float(spearmanr(broad["D"], broad[SCORE_COL]).statistic)
    rho_strict = float(spearmanr(strict["D"], strict[SCORE_COL]).statistic)
    require_close(rho_broad, 0.790796716920716, "Broad n=140 Spearman")
    require_close(rho_strict, 0.8563789152024445, "Strict n=34 Spearman")
    ndcg_broad = ndcg_at_10pct(broad["D"].to_numpy(float), broad[SCORE_COL].to_numpy(float))
    ndcg_strict = ndcg_at_10pct(strict["D"].to_numpy(float), strict[SCORE_COL].to_numpy(float))
    require_close(ndcg_broad, 0.8961863747532142, "Broad n=140 NDCG@10%")
    require_close(ndcg_strict, 0.9395083234807268, "Strict n=34 NDCG@10%")

    frozen_weights = weights_payload["weights_by_member"]
    if len(frozen_weights) != 8 or not np.isclose(sum(frozen_weights.values()), 1.0, atol=1e-12):
        raise AssertionError("Expected a frozen eight-member simplex fusion")

    broad = broad.copy()
    broad["true_D_rank"] = descending_rank(broad["D"])
    broad["predicted_D_rank"] = descending_rank(broad[SCORE_COL])
    broad["true_D_rank_percentile"] = descending_rank_percentile(broad["D"])
    broad["predicted_D_rank_percentile"] = descending_rank_percentile(broad[SCORE_COL])
    broad["absolute_rank_error"] = np.abs(broad["true_D_rank"] - broad["predicted_D_rank"])
    broad["strict_65k_aligned"] = broad["canonical_smiles"].isin(strict_smiles)
    broad["strict_cohort_score"] = broad["canonical_smiles"].map(
        strict.set_index("canonical_smiles")[SCORE_COL]
    )

    # This is diagnostic only: it shows the rank association of strict points
    # after they are embedded in the shared n=140 coordinates.
    strict_common = broad.loc[broad["strict_65k_aligned"]].copy()
    rho_strict_common_coordinates = float(spearmanr(strict_common["D"], strict_common[SCORE_COL]).statistic)
    require_close(rho_strict_common_coordinates, 0.8469060351413291, "Strict overlay Spearman in common n=140 ranking")

    source_columns = [
        "canonical_smiles", "name", "D", SCORE_COL, "strict_cohort_score",
        "true_D_rank", "predicted_D_rank", "absolute_rank_error",
        "true_D_rank_percentile", "predicted_D_rank_percentile", "strict_65k_aligned",
    ]
    broad.loc[:, source_columns].sort_values("true_D_rank_percentile").to_csv(
        SOURCE / f"{STEM}_source_data.csv", index=False
    )

    # Reuse the manuscript Fig. 2c/d visual grammar from 画图代码/新图.py:
    # rank-error colors, a rank-diagonal, and compact in-panel metrics.
    error_bins = [14, 28, 42]  # 10%, 20%, and 30% of the broad n=140 cohort.
    error_styles = [
        (broad["absolute_rank_error"] <= error_bins[0], COLORS["deep_blue"]),
        ((broad["absolute_rank_error"] > error_bins[0]) & (broad["absolute_rank_error"] <= error_bins[1]), COLORS["sky_blue"]),
        ((broad["absolute_rank_error"] > error_bins[1]) & (broad["absolute_rank_error"] <= error_bins[2]), "#b6ccb9"),
        (broad["absolute_rank_error"] > error_bins[2], "#c7b4be"),
    ]
    error_counts = [int(mask.sum()) for mask, _ in error_styles]
    error_percentages = [100.0 * count / len(broad) for count in error_counts]
    fig, ax = plt.subplots(figsize=(5.35, 4.95), constrained_layout=True)
    for mask, color in error_styles:
        ax.scatter(
            broad.loc[mask, "true_D_rank"], broad.loc[mask, "predicted_D_rank"],
            s=28, c=color, edgecolors="white", linewidths=0.38, alpha=0.96, zorder=2,
        )
    ax.plot([1, 140], [1, 140], color=COLORS["mid_gray"], linewidth=1.05, linestyle=(0, (4, 3)), zorder=3)
    # Strict points are nested, so they receive a halo rather than a competing fill color.
    ax.scatter(
        strict_common["true_D_rank"], strict_common["predicted_D_rank"],
        s=53, facecolors="none", edgecolors=COLORS["peach"], linewidths=1.5, alpha=0.98, zorder=4,
    )

    ax.set_xlim(0.5, 140.5)
    ax.set_ylim(0.5, 140.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([1, 20, 60, 100, 140])
    ax.set_yticks([1, 20, 60, 100, 140])
    ax.set_xlabel("True D rank (1 = highest)")
    ax.set_ylabel("Predicted D rank (1 = highest)")
    ax.set_title("External D: Rank Prediction", color=COLORS["ink"], fontweight="bold", pad=7)
    ax.grid(False)

    error_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["deep_blue"], markeredgecolor="white", markeredgewidth=0.4, markersize=5.7, label=f"Excellent (<=10%; n={error_counts[0]}, {error_percentages[0]:.1f}%)"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["sky_blue"], markeredgecolor="white", markeredgewidth=0.4, markersize=5.7, label=f"Good (10-20%; n={error_counts[1]}, {error_percentages[1]:.1f}%)"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#b6ccb9", markeredgecolor="white", markeredgewidth=0.4, markersize=5.7, label=f"Okay (20-30%; n={error_counts[2]}, {error_percentages[2]:.1f}%)"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#c7b4be", markeredgecolor="white", markeredgewidth=0.4, markersize=5.7, label=f"Large error (>30%; n={error_counts[3]}, {error_percentages[3]:.1f}%)"),
        Line2D([0], [0], color=COLORS["mid_gray"], linewidth=1.05, linestyle=(0, (4, 3)), label="Perfect prediction"),
    ]
    first_legend = ax.legend(handles=error_handles, loc="upper left", handlelength=2.15, handletextpad=0.5, borderaxespad=0.5, labelspacing=0.30, fontsize=6.6)
    ax.add_artist(first_legend)
    cohort_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=COLORS["mid_gray"], markeredgewidth=0.85, markersize=6.2,
               label=f"Broad structural-compatible set (n=140)\nSpearman = {rho_broad:.3f}; NDCG@10% = {ndcg_broad:.3f}"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=COLORS["peach"], markeredgewidth=1.45, markersize=6.8,
               label=f"Strict 65k-aligned subset (n=34)\nSpearman = {rho_strict:.3f}; NDCG@10% = {ndcg_strict:.3f}"),
    ]
    ax.legend(handles=cohort_handles, loc="lower right", handlelength=1.8, handletextpad=0.55, borderaxespad=0.55, labelspacing=0.70, fontsize=6.45)

    for suffix, kwargs in (
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(OUT / f"{STEM}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    caption = (
        "External D-rank validation under two pre-specified structural-rule regimes. "
        "Points are colored by absolute rank error following the manuscript Fig. 2c/d rank-prediction convention; peach outlines highlight "
        "the strict 65k-aligned subset (n=34), nested in the broad structural-compatible cohort (n=140). Groups are defined exclusively "
        "by pre-defined structure rules, without D/P labels or prediction-error selection. The reported strict-subset "
        "Spearman correlation is recomputed using the frozen rank fusion within the n=34 cohort; plotted point positions "
        "use common n=140 ranks solely to show the nesting relationship."
    )
    (OUT / f"{STEM}_caption.txt").write_text(caption + "\n", encoding="utf-8")

    manifest = {
        "figure": STEM,
        "purpose": "Independent external D-rank validation panel; it does not replace Fig. 5.",
        "inputs": {
            "broad_predictions": str(BROAD_CSV),
            "strict_predictions": str(STRICT_CSV),
            "frozen_fold0_fusion_weights": str(WEIGHTS_JSON),
        },
        "cohorts": {
            "broad_structural_compatible": {"n": 140, "spearman": rho_broad, "ndcg_at_10pct": ndcg_broad},
            "strict_65k_aligned": {"n": 34, "spearman": rho_strict, "ndcg_at_10pct": ndcg_strict, "strict_is_subset_of_broad": True},
        },
        "display": {
            "coordinates": "Observed and predicted D ranks (1 = highest) are both recomputed across the common n=140 broad cohort.",
            "strict_overlay_spearman_in_common_coordinates": rho_strict_common_coordinates,
            "strict_metric_note": "The formal strict n=34 rho uses frozen rank fusion recomputed within n=34; it is reported in the legend and metric table.",
            "identity_line": "y=x (perfect rank agreement)",
            "fit_lines": "None",
            "rank_error_bins": {"excellent": "<=10% (<=14 ranks)", "good": "10-20% (15-28 ranks)", "okay": "20-30% (29-42 ranks)", "large_error": ">30% (>42 ranks)"},
            "rank_error_distribution": {"excellent": {"n": error_counts[0], "fraction": error_percentages[0] / 100.0}, "good": {"n": error_counts[1], "fraction": error_percentages[1] / 100.0}, "okay": {"n": error_counts[2], "fraction": error_percentages[2] / 100.0}, "large_error": {"n": error_counts[3], "fraction": error_percentages[3] / 100.0}},
            "palette": COLORS,
        },
        "integrity_checks": {
            "broad_unique_smiles": int(broad["canonical_smiles"].nunique()),
            "strict_unique_smiles": int(strict["canonical_smiles"].nunique()),
            "strict_subset_of_broad": strict_smiles.issubset(broad_smiles),
            "frozen_member_count": len(frozen_weights),
            "frozen_weight_sum": float(sum(frozen_weights.values())),
            "grouping_uses_labels_or_prediction_error": False,
        },
        "caption": caption,
    }
    (OUT / f"{STEM}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"broad n={len(broad)} rho={rho_broad:.8f}")
    print(f"strict n={len(strict)} rho={rho_strict:.8f}")
    print(f"strict overlay rho in shared n=140 coordinates={rho_strict_common_coordinates:.8f}")
    print(f"wrote {OUT / (STEM + '.png')}")


if __name__ == "__main__":
    main()
