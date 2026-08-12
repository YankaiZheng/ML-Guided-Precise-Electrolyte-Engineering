#!/usr/bin/env python3
"""Paper-main interpretability figure for ensemble-consensus reliability."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs"
INPUT = RUN / "domain65k_d_candidate100_fold0_fusion_multiobjective/evaluation4000_predictions.csv"
OUT = ROOT / "analysis_outputs/paper_figures_ml_workflow"
SOURCE = OUT / "source_data"
STEM = OUT / "Fig_interpretability_consensus_reliability"
SEED = 20260719

COLORS = {
    "deep_blue": "#4f779e",
    "sky_blue": "#91b4d1",
    "green": "#b6ccb9",
    "peach": "#e1aca6",
    "ink": "#263238",
    "mid_gray": "#7b8790",
    "light_gray": "#dce3e7",
    "near_white": "#f4f7f9",
}

MEMBER_COLUMNS = [
    "score__multiobjective__cat_rank",
    "score__multiobjective__cat_raw",
    "score__multiobjective__lgbm_normal",
    "score__multiobjective__lgbm_rank",
    "score__multiobjective__lgbm_raw",
    "score__multiobjective__xgb_rank",
    "score__multiobjective__xgb_raw",
    "score__multiobjective__vector_mu",
]


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.titlesize": 8.1,
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.fontsize": 6.4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top")


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = 2000):
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        values[i] = spearmanr(x[idx], y[idx]).statistic
    return np.quantile(values, [0.025, 0.975])


def cumulative_curve(
    frame: pd.DataFrame,
    value: str,
    coverages: np.ndarray,
    statistic: str,
) -> np.ndarray:
    ordered = frame.sort_values("member_rank_iqr")
    output = []
    for coverage in coverages:
        n_keep = max(1, int(round(len(ordered) * coverage)))
        values = ordered.iloc[:n_keep][value]
        output.append(float(values.mean() if statistic == "mean" else values.median()))
    return np.asarray(output)


def bootstrap_curve(
    frame: pd.DataFrame,
    value: str,
    coverages: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
    n_boot: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    curves = np.empty((n_boot, len(coverages)), dtype=float)
    values = frame.to_numpy()
    columns = list(frame.columns)
    for i in range(n_boot):
        sampled = pd.DataFrame(values[rng.integers(0, len(frame), len(frame))], columns=columns)
        curves[i] = cumulative_curve(sampled, value, coverages, statistic)
    return np.quantile(curves, 0.025, axis=0), np.quantile(curves, 0.975, axis=0)


def build_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(INPUT)
    if len(df) != 4000 or df[MEMBER_COLUMNS].isna().any().any():
        raise RuntimeError("Expected complete eight-member predictions for 4,000 evaluation molecules")

    member_ranks = pd.DataFrame({
        column: df[column].rank(method="average", ascending=False, pct=True) * 100
        for column in MEMBER_COLUMNS
    })
    disagreement = member_ranks.quantile(0.75, axis=1) - member_ranks.quantile(0.25, axis=1)
    true_rank = df["D"].rank(method="average", ascending=False, pct=True) * 100
    fusion_rank = df["score__final_fusion_multiobjective"].rank(
        method="average", ascending=False, pct=True
    ) * 100
    points = pd.DataFrame({
        "locked_row_id": df["locked_row_id"],
        "domain_index": df["domain_index"],
        "canonical_smiles": df["canonical_smiles"],
        "D": df["D"],
        "true_rank_percentile": true_rank,
        "fusion_rank_percentile": fusion_rank,
        "member_rank_iqr": disagreement,
        "absolute_rank_error": (fusion_rank - true_rank).abs(),
        "predicted_top10": fusion_rank <= 10,
        "true_top10": true_rank <= 10,
    })

    quintile = pd.qcut(points["member_rank_iqr"], 5, labels=False, duplicates="drop")
    rng = np.random.default_rng(SEED)
    bin_rows = []
    for index, group in points.groupby(quintile, observed=True):
        medians = np.empty(2000, dtype=float)
        values = group["absolute_rank_error"].to_numpy(float)
        for i in range(len(medians)):
            medians[i] = np.median(values[rng.integers(0, len(values), len(values))])
        bin_rows.append({
            "disagreement_quintile": int(index) + 1,
            "n": len(group),
            "median_member_rank_iqr": group["member_rank_iqr"].median(),
            "median_absolute_rank_error": np.median(values),
            "ci_low": np.quantile(medians, 0.025),
            "ci_high": np.quantile(medians, 0.975),
        })
    bins = pd.DataFrame(bin_rows)

    coverages = np.linspace(0.10, 1.00, 19)
    risk = cumulative_curve(points, "absolute_rank_error", coverages, "mean")
    risk_low, risk_high = bootstrap_curve(
        points[["member_rank_iqr", "absolute_rank_error"]],
        "absolute_rank_error", coverages, "mean", rng,
    )
    selected = points.loc[points["predicted_top10"], ["member_rank_iqr", "true_top10"]].rename(
        columns={"true_top10": "hit"}
    )
    precision = cumulative_curve(selected, "hit", coverages, "mean")
    precision_low, precision_high = bootstrap_curve(selected, "hit", coverages, "mean", rng)
    curves = pd.DataFrame({
        "coverage": coverages,
        "coverage_percent": coverages * 100,
        "mean_absolute_rank_error": risk,
        "risk_ci_low": risk_low,
        "risk_ci_high": risk_high,
        "top10_precision": precision,
        "precision_ci_low": precision_low,
        "precision_ci_high": precision_high,
    })

    rho = float(spearmanr(points["member_rank_iqr"], points["absolute_rank_error"]).statistic)
    rho_ci = bootstrap_spearman(
        points["member_rank_iqr"].to_numpy(float),
        points["absolute_rank_error"].to_numpy(float), rng,
    )
    summary = {
        "n_evaluation": len(points),
        "n_members": len(MEMBER_COLUMNS),
        "disagreement_definition": "IQR of eight member rank percentiles",
        "error_definition": "absolute difference between frozen fusion and reference D rank percentiles",
        "spearman_disagreement_error": rho,
        "spearman_bootstrap_95ci": [float(rho_ci[0]), float(rho_ci[1])],
        "predicted_top10_n": int(points["predicted_top10"].sum()),
        "top10_precision_at_10pct_consensus_coverage": float(curves.iloc[0]["top10_precision"]),
        "top10_precision_at_full_coverage": float(curves.iloc[-1]["top10_precision"]),
    }
    return points, bins, curves, summary


def plot(points: pd.DataFrame, bins: pd.DataFrame, curves: pd.DataFrame, summary: dict) -> None:
    configure_style()
    density_cmap = ListedColormap([
        "#edf3f7", "#d8e5ed", "#b9cfdd", "#83a9c2", COLORS["deep_blue"]
    ])
    fig, ax = plt.subplots(figsize=(4.75, 3.45))

    ax.hexbin(
        points["member_rank_iqr"], points["absolute_rank_error"],
        gridsize=(34, 27), mincnt=1, bins="log", cmap=density_cmap,
        linewidths=0, extent=(0, 32, 0, 80), rasterized=True,
    )
    ax.errorbar(
        bins["median_member_rank_iqr"], bins["median_absolute_rank_error"],
        yerr=[
            bins["median_absolute_rank_error"] - bins["ci_low"],
            bins["ci_high"] - bins["median_absolute_rank_error"],
        ],
        color=COLORS["ink"], marker="o", markersize=4.2, linewidth=1.45,
        capsize=2.2, markerfacecolor="white", markeredgewidth=1.0, zorder=5,
        label="Quintile median (95% CI)",
    )
    rho = summary["spearman_disagreement_error"]
    ci = summary["spearman_bootstrap_95ci"]
    ax.text(
        0.97, 0.96,
        rf"$\rho$ = {rho:.2f}  (95% CI {ci[0]:.2f}–{ci[1]:.2f})" + "\n" + r"$n$ = 4,000",
        transform=ax.transAxes, ha="right", va="top", color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.2},
    )
    ax.set(
        xlim=(0, 32), ylim=(0, 80),
        xlabel="Eight-member rank disagreement (IQR, percentile points)",
        ylabel="Absolute fusion-rank error (percentile points)",
    )
    ax.set_title("Ensemble disagreement identifies unreliable rankings", loc="left", fontweight="bold", pad=7)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.84), handlelength=1.6, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.92)
    ax.annotate(
        "Higher disagreement",
        xy=(0.96, 0.035), xytext=(0.72, 0.035), xycoords="axes fraction",
        ha="center", va="center", color=COLORS["mid_gray"], fontsize=6.5,
        arrowprops={"arrowstyle": "-|>", "color": COLORS["mid_gray"], "lw": 0.8},
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.18, top=0.90)

    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    points, bins, curves, summary = build_source_data()
    points.to_csv(SOURCE / "interpretability_consensus_points.csv", index=False)
    bins.to_csv(SOURCE / "interpretability_consensus_quintiles.csv", index=False)
    curves.to_csv(SOURCE / "interpretability_consensus_curves.csv", index=False)
    plot(points, bins, curves, summary)
    manifest = {
        "surface": "paper_main",
        "core_conclusion": "Eight-member rank disagreement is positively associated with frozen-fusion ranking error.",
        "archetype": "single-panel quantitative relationship",
        "backend": "Python/matplotlib",
        "final_size_inches": [4.75, 3.45],
        "evaluation_cohort": "frozen chemistry-curated 4,000-molecule evaluation set",
        "model_policy": "all predictions and fusion weights frozen before this post-hoc interpretability analysis",
        "statistics": summary,
        "exports": [str(STEM.with_suffix(ext)) for ext in (".svg", ".pdf", ".png", ".tiff")],
        "source_data": [
            str(SOURCE / "interpretability_consensus_points.csv"),
            str(SOURCE / "interpretability_consensus_quintiles.csv"),
        ],
        "image_integrity": "No molecular images or selective image adjustments; all marks are generated from tabular predictions.",
        "reviewer_risk": "Consensus is an internal reliability score evaluated post hoc, not a calibrated probabilistic uncertainty and not used to retune the model.",
    }
    (OUT / "Fig_interpretability_consensus_reliability_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
