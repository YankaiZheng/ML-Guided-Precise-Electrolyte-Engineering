#!/usr/bin/env python3
"""Plot the formal vector-model training and low-LR continuation histories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs"
MAIN = RUNS / "domain65k_d_vector_candidate100_gpu_fold0"
LOW = RUNS / "domain65k_d_vector_candidate100_gpu_fold0_lr5e5_e35"
OUT = ROOT / "analysis_outputs/paper_figures_ml_workflow"
SOURCE = OUT / "source_data"

COLORS = {
    "deep_blue": "#4F779E",
    "sky_blue": "#91B4D1",
    "green": "#B6CCB9",
    "deep_green": "#88A98E",
    "peach": "#E1ACA6",
    "ink": "#263238",
    "mid_gray": "#7B8790",
    "light_gray": "#DCE3E7",
    "near_white": "#F4F7F9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage(run: Path, stage: str, epoch_offset: int, learning_rate: float) -> pd.DataFrame:
    history_path = run / "fold0_history.csv"
    metrics_path = run / "fold0_epoch_metrics.csv"
    history = pd.read_csv(history_path)
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[
        metrics["member"].eq("pred__vector_mu"),
        ["epoch", "spearman", "ndcg_at_10pct", "rmse", "mae_calibrated"],
    ]
    frame = history.merge(metrics, on="epoch", how="left", validate="one_to_one")
    frame.insert(0, "record_epoch", frame["epoch"].astype(int) + epoch_offset)
    frame.insert(1, "stage", stage)
    frame.insert(2, "learning_rate", learning_rate)
    frame = frame.rename(
        columns={
            "epoch": "stage_epoch",
            "loss": "train_loss",
            "inner_mu_spearman": "history_spearman",
        }
    )
    # The history Spearman is available from Epoch 1, whereas the detailed
    # metric logger was added at Epoch 5. They are the same vector-magnitude
    # statistic wherever both exist.
    overlap = frame["spearman"].notna()
    if not np.allclose(
        frame.loc[overlap, "history_spearman"],
        frame.loc[overlap, "spearman"],
        atol=1e-12,
    ):
        raise RuntimeError(f"Spearman mismatch between history and metrics in {run}")
    frame["spearman"] = frame["spearman"].fillna(frame["history_spearman"])
    return frame[
        [
            "record_epoch",
            "stage",
            "stage_epoch",
            "learning_rate",
            "train_loss",
            "spearman",
            "ndcg_at_10pct",
            "rmse",
            "mae_calibrated",
            "inner_rank_spearman",
            "selection_score",
            "elapsed_sec",
        ]
    ]


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        colors=COLORS["ink"],
        labelsize=8.1,
        direction="out",
        length=3.2,
        width=0.7,
    )
    ax.grid(False)
    ax.axvspan(50.5, 54.5, color=COLORS["peach"], alpha=0.12, zorder=0)
    ax.axvline(50.5, color=COLORS["mid_gray"], lw=0.8, ls=(0, (3, 3)), zorder=1)


def plot_segment(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    color: str,
    marker: str,
    zorder: int,
) -> None:
    valid = frame[column].notna()
    ax.plot(
        frame.loc[valid, "record_epoch"],
        frame.loc[valid, column],
        color=color,
        lw=1.45,
        marker=marker,
        markersize=3.1,
        markeredgewidth=0,
        zorder=zorder,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    main_frame = load_stage(MAIN, "High LR", 0, 3e-4)
    low_frame = load_stage(LOW, "Low LR branch", 50, 5e-5)
    frame = pd.concat([main_frame, low_frame], ignore_index=True)

    if len(main_frame) != 50 or len(low_frame) != 4 or len(frame) != 54:
        raise RuntimeError(
            f"Unexpected epoch counts: main={len(main_frame)}, low={len(low_frame)}"
        )
    if main_frame["record_epoch"].tolist() != list(range(1, 51)):
        raise RuntimeError("Main-stage epoch order is incomplete")
    if low_frame["record_epoch"].tolist() != list(range(51, 55)):
        raise RuntimeError("Low-LR record order is incomplete")

    source_path = SOURCE / "Fig_vector_training_54epoch_source_data.csv"
    frame.to_csv(source_path, index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.15, 6.45),
        sharex=True,
        gridspec_kw={"height_ratios": [1.18, 1.0, 1.0, 1.0], "hspace": 0.17},
    )

    panels = [
        ("train_loss", "Training loss", COLORS["deep_blue"]),
        ("rmse", "Validation RMSE", COLORS["mid_gray"]),
        ("spearman", "Spearman", COLORS["deep_blue"]),
        ("ndcg_at_10pct", "NDCG@10%", COLORS["deep_green"]),
    ]
    for panel, (column, ylabel, high_color) in zip(axes, panels):
        style_axis(panel)
        plot_segment(panel, main_frame, column, high_color, "o", 3)
        plot_segment(panel, low_frame, column, COLORS["peach"], "o", 4)
        panel.set_ylabel(ylabel, fontsize=9.1, fontweight="bold", labelpad=8)

        frozen = low_frame.loc[low_frame["stage_epoch"].eq(1)].iloc[0]
        if pd.notna(frozen[column]):
            panel.scatter(
                [frozen["record_epoch"]],
                [frozen[column]],
                s=47,
                facecolor=COLORS["peach"],
                edgecolor=COLORS["deep_blue"],
                linewidth=1.25,
                zorder=7,
            )

    axes[0].set_ylim(0.12, 2.32)
    axes[0].set_yticks([0.2, 0.5, 1.0, 1.5, 2.0])
    loss_inset = axes[0].inset_axes([0.22, 0.43, 0.34, 0.47])
    plot_segment(
        loss_inset,
        main_frame.loc[main_frame["stage_epoch"].ge(2)],
        "train_loss",
        COLORS["deep_blue"],
        "o",
        3,
    )
    plot_segment(
        loss_inset,
        low_frame,
        "train_loss",
        COLORS["peach"],
        "o",
        4,
    )
    loss_inset.axvspan(50.5, 54.5, color=COLORS["peach"], alpha=0.12, zorder=0)
    loss_inset.axvline(
        50.5, color=COLORS["mid_gray"], lw=0.65, ls=(0, (3, 3)), zorder=1
    )
    loss_inset.scatter(
        [51],
        [low_frame.loc[low_frame["stage_epoch"].eq(1), "train_loss"].iloc[0]],
        s=24,
        facecolor=COLORS["peach"],
        edgecolor=COLORS["deep_blue"],
        linewidth=0.9,
        zorder=7,
    )
    loss_inset.set_xlim(1.5, 54.5)
    loss_inset.set_ylim(0.15, 0.47)
    loss_inset.set_xticks([2, 20, 35, 50, 54])
    loss_inset.set_yticks([0.2, 0.3, 0.4])
    loss_inset.tick_params(
        axis="both",
        labelsize=6.2,
        direction="out",
        length=2.3,
        width=0.55,
        colors=COLORS["ink"],
    )
    for spine in loss_inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(COLORS["mid_gray"])
    loss_inset.set_title(
        "Epochs 2–54 (linear zoom)",
        fontsize=6.5,
        color=COLORS["ink"],
        pad=2.5,
    )

    axes[1].set_ylim(0.90, 1.045)
    axes[1].set_yticks([0.92, 0.96, 1.00, 1.04])

    axes[2].set_ylim(0.62, 0.795)
    axes[2].set_yticks([0.65, 0.70, 0.75, 0.78])

    axes[3].set_ylim(0.875, 0.916)
    axes[3].set_yticks([0.88, 0.89, 0.90, 0.91])
    axes[3].set_xlim(0.5, 54.5)
    axes[3].set_xticks([1, 10, 20, 30, 35, 40, 50, 54])
    axes[3].set_xlabel("Recorded epoch", fontsize=9.4, fontweight="bold")

    axes[0].text(
        0.01,
        0.91,
        "Epoch-mean training objective",
        transform=axes[0].transAxes,
        color=COLORS["mid_gray"],
        fontsize=7.0,
        va="top",
    )
    axes[3].text(
        0.012,
        0.08,
        "Detailed validation metrics logged from Epoch 5",
        transform=axes[3].transAxes,
        color=COLORS["mid_gray"],
        fontsize=7.0,
    )
    for ax in (axes[1], axes[3]):
        ax.text(
            0.012,
            0.92,
            "Inner validation",
            transform=ax.transAxes,
            color=COLORS["mid_gray"],
            fontsize=7.0,
            va="top",
        )
    axes[1].text(
        0.012,
        0.81,
        "Stored validation error; not composite validation loss",
        transform=axes[1].transAxes,
        color=COLORS["mid_gray"],
        fontsize=6.6,
        va="top",
    )
    axes[2].text(
        0.012,
        0.92,
        "Inner validation",
        transform=axes[2].transAxes,
        color=COLORS["mid_gray"],
        fontsize=7.0,
        va="top",
    )
    for label, ax in zip(("a", "b", "c", "d"), axes):
        ax.text(
            -0.075,
            1.01,
            label,
            transform=ax.transAxes,
            fontsize=9.2,
            fontweight="bold",
            color=COLORS["ink"],
            ha="right",
            va="bottom",
        )

    source_epoch = main_frame.loc[main_frame["stage_epoch"].eq(35)].iloc[0]
    branch_epoch = low_frame.loc[low_frame["stage_epoch"].eq(1)].iloc[0]
    axes[0].annotate(
        "Low-LR branch initialized\nfrom Epoch 35 checkpoint",
        xy=(35, source_epoch["train_loss"]),
        xytext=(38.7, 1.12),
        fontsize=7.4,
        color=COLORS["ink"],
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["mid_gray"],
            "lw": 0.9,
            "connectionstyle": "arc3,rad=-0.12",
        },
    )
    axes[0].annotate(
        "Frozen checkpoint",
        xy=(51, branch_epoch["train_loss"]),
        xytext=(48.7, 0.50),
        fontsize=7.3,
        color=COLORS["deep_blue"],
        ha="right",
        va="bottom",
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["deep_blue"],
            "lw": 0.8,
        },
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["deep_blue"],
            marker="o",
            lw=1.5,
            markersize=4,
            label=r"High LR ($3\times10^{-4}$; 50 records)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["peach"],
            marker="o",
            lw=1.5,
            markersize=4,
            label=r"Low LR ($5\times10^{-5}$; 4 records)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["peach"],
            markeredgecolor=COLORS["deep_blue"],
            markeredgewidth=1.1,
            markersize=6,
            label="Frozen checkpoint",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.982, 0.965),
        ncol=3,
        frameon=False,
        fontsize=7.2,
        handlelength=1.8,
        columnspacing=1.4,
        handletextpad=0.55,
    )

    fig.suptitle(
        "Vector-model training dynamics",
        x=0.12,
        y=0.995,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.subplots_adjust(left=0.125, right=0.985, top=0.875, bottom=0.09)

    base = OUT / "Fig_vector_training_54epoch"
    fig.savefig(base.with_suffix(".png"), dpi=360, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    manifest = {
        "title": "Vector-model training dynamics",
        "surface_class": "paper_main",
        "source_files": {
            "main_history": str(MAIN / "fold0_history.csv"),
            "main_epoch_metrics": str(MAIN / "fold0_epoch_metrics.csv"),
            "low_lr_history": str(LOW / "fold0_history.csv"),
            "low_lr_epoch_metrics": str(LOW / "fold0_epoch_metrics.csv"),
        },
        "source_sha256": {
            "main_history": sha256(MAIN / "fold0_history.csv"),
            "main_epoch_metrics": sha256(MAIN / "fold0_epoch_metrics.csv"),
            "low_lr_history": sha256(LOW / "fold0_history.csv"),
            "low_lr_epoch_metrics": sha256(LOW / "fold0_epoch_metrics.csv"),
        },
        "source_data": str(source_path),
        "n_recorded_epochs": 54,
        "trajectory_note": (
            "The four low-LR records form a branch initialized from the main "
            "run's Epoch 35 best checkpoint; they are displayed at record "
            "positions 51-54 and are not connected to Epoch 50."
        ),
        "metric_note": (
            "Spearman and NDCG@10% are inner-validation metrics for the "
            "vector-magnitude prediction. Detailed NDCG logging began at "
            "main-run Epoch 5; no values were imputed for Epochs 1-4."
        ),
        "frozen_checkpoint": {
            "stage": "low_lr",
            "stage_epoch": 1,
            "record_epoch": 51,
            "train_loss": float(branch_epoch["train_loss"]),
            "spearman": float(branch_epoch["spearman"]),
            "ndcg_at_10pct": float(branch_epoch["ndcg_at_10pct"]),
        },
        "exports": {
            suffix: str(base.with_suffix(suffix))
            for suffix in [".png", ".pdf", ".svg", ".tiff"]
        },
    }
    (OUT / "Fig_vector_training_54epoch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
