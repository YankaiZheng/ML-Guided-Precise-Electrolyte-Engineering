#!/usr/bin/env python3
"""Build the data-driven Pareto-knee selection panel for the frozen 78 candidates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "candidates"
SOURCE = ROOT / "data" / "figure_source"
OUT = ROOT / "results" / "figures"

CANDIDATES_CSV = DATA / "candidate78_final_predictions.csv"
KNEE_CSV = DATA / "candidate78_pareto_knee.csv"
STEM = "Fig_candidate78_pareto_knee_selection"

COLORS = {
    "deep_blue": "#4f779e",
    "sky_blue": "#91b4d1",
    "peach": "#e1aca6",
    "ink": "#263238",
    "mid_gray": "#7b8790",
    "light_gray": "#dce3e7",
    "near_white": "#f4f7f9",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.8,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.5,
            "legend.fontsize": 7.1,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def point_to_line_projection(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    chord = end - start
    t = float(np.dot(point - start, chord) / np.dot(chord, chord))
    projection = start + t * chord
    return projection, float(np.linalg.norm(point - projection))


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(CANDIDATES_CSV)
    knee = pd.read_csv(KNEE_CSV)
    if len(candidates) != 78 or candidates["canonical_smiles"].nunique() != 78:
        raise AssertionError("Expected 78 unique frozen candidates")
    if len(knee) != 8 or knee["selected"].sum() != 1:
        raise AssertionError("Expected an eight-point Pareto front with one selected knee")
    if knee.loc[knee["selected"], "standard_name"].item() != "DMTMSA":
        raise AssertionError("DMTMSA must be the official selected knee")

    # The compact source-data copies are deliberately checked against the
    # released candidate tables before any panel is rendered.
    published_candidates = pd.read_csv(SOURCE / "Fig_candidate78_pareto_knee_selection_candidate_source_data.csv")
    published_pareto = pd.read_csv(SOURCE / "Fig_candidate78_pareto_knee_selection_pareto_source_data.csv")
    if len(published_candidates) != len(candidates) or len(published_pareto) != len(knee):
        raise AssertionError("Published Pareto figure source data are incomplete")
    source_names = set(published_candidates["canonical_smiles"])
    candidate_names = set(candidates["canonical_smiles"])
    if source_names != candidate_names:
        raise AssertionError("Published Pareto figure source data do not match the candidate release")
    source_knee = published_pareto.set_index("standard_name")["knee_distance"]
    frozen_knee = knee.set_index("standard_name")["knee_distance"]
    if set(source_knee.index) != set(frozen_knee.index) or not np.allclose(source_knee.sort_index(), frozen_knee.sort_index(), atol=1e-6):
        raise AssertionError("Published Pareto figure source distances do not match the frozen selection table")

    candidates = candidates.copy()
    n = len(candidates)
    candidates["qD"] = (n - candidates["D_rank_final78"].astype(float)) / (n - 1)
    candidates["qP"] = (n - candidates["P_rank_final78"].astype(float)) / (n - 1)
    candidates["pareto"] = candidates["D_P_pareto"].astype(bool)

    candidate_pareto_coords = candidates.loc[candidates["pareto"], ["qD", "qP"]].to_numpy()
    knee_coords = knee[["qD", "qP"]].to_numpy()
    candidate_pareto_coords = candidate_pareto_coords[np.lexsort((candidate_pareto_coords[:, 1], candidate_pareto_coords[:, 0]))]
    knee_coords = knee_coords[np.lexsort((knee_coords[:, 1], knee_coords[:, 0]))]
    if not np.allclose(candidate_pareto_coords, knee_coords, atol=1e-12):
        raise AssertionError("The eight frozen Pareto coordinates do not match the candidate table")

    pareto = knee.sort_values("qP").copy()
    endpoints = pareto.iloc[[0, -1]].copy()
    start = endpoints.iloc[0][["qP", "qD"]].to_numpy(dtype=float)
    end = endpoints.iloc[1][["qP", "qD"]].to_numpy(dtype=float)
    projections = []
    for row in pareto.itertuples(index=False):
        projection, distance = point_to_line_projection(np.array([row.qP, row.qD]), start, end)
        projections.append((projection[0], projection[1], distance))
    pareto[["chord_qP", "chord_qD", "recomputed_knee_distance"]] = projections
    if not np.allclose(pareto["recomputed_knee_distance"], pareto["knee_distance"], atol=1e-12):
        raise AssertionError("Knee distances do not reproduce frozen official table")

    dmt = pareto.loc[pareto["selected"]].iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    dominated = candidates.loc[~candidates["pareto"]]
    ax.scatter(
        dominated["qP"], dominated["qD"], s=27, c=COLORS["light_gray"],
        edgecolors="white", linewidths=0.35, alpha=1.0, zorder=1,
    )
    ax.plot(pareto["qP"], pareto["qD"], color=COLORS["deep_blue"], linewidth=1.55, zorder=3)
    ax.plot([start[0], end[0]], [start[1], end[1]], color=COLORS["mid_gray"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)

    # Keep all subordinate knee distances thin; DMTMSA carries the only accent.
    for row in pareto.loc[~pareto["selected"]].itertuples(index=False):
        if row.knee_distance <= 1e-12:
            continue
        ax.plot([row.qP, row.chord_qP], [row.qD, row.chord_qD], color=COLORS["mid_gray"], linewidth=0.72, alpha=0.8, zorder=2)
    ax.plot([dmt.qP, dmt.chord_qP], [dmt.qD, dmt.chord_qD], color=COLORS["peach"], linewidth=2.05, zorder=4)

    regular = pareto.loc[~pareto["selected"]]
    ax.scatter(regular["qP"], regular["qD"], s=49, c=COLORS["deep_blue"], edgecolors="white", linewidths=0.55, zorder=4)
    ax.scatter([dmt.qP], [dmt.qD], s=92, c=COLORS["peach"], edgecolors=COLORS["deep_blue"], linewidths=1.35, zorder=5)

    ax.annotate(
        "DMTMSA", xy=(dmt.qP, dmt.qD), xytext=(dmt.qP - 0.10, dmt.qD + 0.055),
        color=COLORS["ink"], fontsize=8.6, fontweight="bold", ha="right", va="bottom",
        arrowprops={"arrowstyle": "-", "color": COLORS["ink"], "linewidth": 0.72, "shrinkA": 3, "shrinkB": 5},
        zorder=6,
    )
    ax.annotate("D extreme", xy=(start[0], start[1]), xytext=(start[0] - 0.02, start[1] + 0.055),
                color=COLORS["mid_gray"], fontsize=7.0, ha="right", va="bottom")
    ax.annotate("P extreme", xy=(end[0], end[1]), xytext=(end[0] + 0.025, end[1] - 0.055),
                color=COLORS["mid_gray"], fontsize=7.0, ha="left", va="top")
    ax.annotate(
        "maximum knee\ndistance", xy=((dmt.qP + dmt.chord_qP) / 2, (dmt.qD + dmt.chord_qD) / 2),
        xytext=(0.59, 0.56), color="#c76f5f", fontsize=7.2, ha="center", va="center",
        arrowprops={"arrowstyle": "-", "color": COLORS["peach"], "linewidth": 0.65, "shrinkA": 2, "shrinkB": 2},
        zorder=6,
    )

    ax.set_xlim(-0.035, 1.045)
    ax.set_ylim(-0.035, 1.045)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Normalized P rank (higher P = 1)")
    ax.set_ylabel("Normalized D rank (higher D = 1)")
    ax.set_title("Data-driven Pareto-knee selection of DMTMSA", color=COLORS["deep_blue"], fontweight="bold", pad=9)
    ax.grid(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["light_gray"], markeredgecolor="white", markeredgewidth=0.4, markersize=5.6,
               label="Pareto-dominated candidates (n=70)"),
        Line2D([0], [0], marker="o", linestyle="-", color=COLORS["deep_blue"], markerfacecolor=COLORS["deep_blue"], markeredgecolor="white", markeredgewidth=0.5, markersize=5.7,
               label="Pareto frontier (n=8)"),
        Line2D([0], [0], color=COLORS["mid_gray"], linewidth=1.0, linestyle=(0, (4, 3)), label="Endpoint chord"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["peach"], markeredgecolor=COLORS["deep_blue"], markeredgewidth=1.0, markersize=6.7,
               label="Maximum-distance knee"),
    ]
    ax.legend(handles=handles, loc="lower left", handlelength=2.1, handletextpad=0.55, borderaxespad=0.7, labelspacing=0.48)

    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"{STEM}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    caption = (
        "Data-driven selection among the 78 electrolyte candidates. D and P ranks are independently normalized to [0, 1], "
        "and Pareto-dominated candidates are removed before evaluation of the eight-point frontier. The endpoint chord connects "
        "the P-extreme and D-extreme frontier members. DMTMSA has the largest perpendicular distance to this chord "
        f"(d = {dmt.knee_distance:.3f}), identifying it as the maximum-knee compromise between D and P."
    )
    (OUT / f"{STEM}_caption.txt").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "figure": STEM,
        "inputs": {"candidate_predictions": str(CANDIDATES_CSV), "frozen_knee_table": str(KNEE_CSV)},
        "n_candidates": 78,
        "n_pareto": 8,
        "n_dominated": 70,
        "selected": {"standard_name": "DMTMSA", "knee_distance": float(dmt.knee_distance)},
        "normalization": "q = (78 - rank) / 77; rank 1 maps to 1 and rank 78 maps to 0.",
        "selection_rule": "Among precomputed Pareto-nondominated candidates, select the maximum perpendicular distance to the chord joining the two frontier endpoints.",
        "caption": caption,
    }
    (OUT / f"{STEM}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DMTMSA knee distance = {dmt.knee_distance:.6f}")
    print(f"wrote {OUT / (STEM + '.png')}")


if __name__ == "__main__":
    main()
