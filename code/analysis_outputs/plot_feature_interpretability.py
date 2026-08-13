#!/usr/bin/env python3
"""Plot the paper-facing weighted TreeSHAP feature interpretation figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "figure_source"
OUT = ROOT / "results" / "figures"
STEM = OUT / "Fig_interpretability_feature_treeshap"

INK = "#27343B"
MUTED = "#71808A"
GRID = "#DCE3E6"
FAMILY_COLORS = {
    "2D descriptors": "#7F9FB9",
    "Three-conformer xTB": "#4F779E",
    "Atom-resolved 3D": "#B6CCB9",
    "Single-conformer xTB": "#E1ACA6",
    "Dipole-derived": "#8E99A2",
}
DISPLAY = {
    "mc3__xtb_mc3_xtb_full_dipole_debye_mean": "Mean μxTB (3C)",
    "cheap__v3__rd2d_fr_nitrile": "Nitrile #",
    "atom3d__full_mu_mean": "μAR",
    "cheap__v4__v4_mmffq3d_dipole_norm_lowE": "LE qμMMFF",
    "mc3__xtb_mc3_xtb_direct_charge_dipole_norm_mean": "Mean qμ (3C)",
    "single__xtb_direct_charge_dipole_norm": "qμ (1C)",
    "single__xtb_full_dipole_debye": "μxTB (1C)",
    "mc3__xtb_mc3_xtb_full_dipole_debye_min": "Min μxTB (3C)",
    "mc3__xtb_mc3_xtb_full_dipole_debye_max": "Max μxTB (3C)",
    "cheap__v3__rd3d_getaway_264": "GET264",
    "mc3__xtb_mc3_xtb_full_dipole_debye_lowe": "LE μxTB (3C)",
    "cheap__v5__v5_mmffq_rbin0_heavy_abs_qprod": "QprodMMFF",
    "atom3d__full_vector_resultant": "μres",
    "cheap__v5__v5_gast_rbin0_heavy_abs_qprod": "QprodGast",
    "derived__mu_per_mw": "μxTB / MW",
}
ABBREVIATION_KEY = {
    "μ": "dipole magnitude",
    "qμ": "charge-only dipole magnitude",
    "1C / 3C": "single-conformer / three-conformer statistic",
    "LE": "lowest-energy conformer",
    "AR": "atom-resolved",
    "MW": "molecular mass",
    "Qprod": "local absolute-charge product descriptor",
}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 11.2,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
    })


def swarm_offsets(values: np.ndarray, row: int, rng: np.random.Generator) -> np.ndarray:
    edges = np.linspace(-0.68, 0.55, 78)
    bins = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
    offsets = np.zeros(len(values), dtype=float)
    max_count = max(int(np.sum(bins == index)) for index in np.unique(bins))
    spacing = 0.62 / max(max_count, 1)
    for bin_index in np.unique(bins):
        indices = np.flatnonzero(bins == bin_index)
        rng.shuffle(indices)
        count = len(indices)
        centered = (np.arange(count) - (count - 1) / 2.0) * spacing
        offsets[indices] = np.clip(centered, -0.31, 0.31)
    return row + offsets


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(SOURCE / "feature_interpretability_top15_points.csv")
    global_importance = pd.read_csv(SOURCE / "feature_interpretability_global_importance.csv")
    family = pd.read_csv(SOURCE / "feature_interpretability_family_importance.csv")
    ordered = global_importance.head(15)["feature"].astype(str).tolist()
    if set(ordered) != set(points["feature"].astype(str)):
        raise RuntimeError("Top-feature source tables are inconsistent")

    cmap = LinearSegmentedColormap.from_list("feature_value", ["#DDA69F", "#F2F2ED", "#4F779E"])
    fig = plt.figure(figsize=(7.15, 5.35), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[0.88, 5.4], hspace=0.08)
    family_ax = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])

    family_order = [
        "2D descriptors",
        "Three-conformer xTB",
        "Atom-resolved 3D",
        "Single-conformer xTB",
        "Dipole-derived",
    ]
    family_share = family.set_index("family")["importance_fraction"].to_dict()
    left = 0.0
    for name in family_order:
        share = float(family_share.get(name, 0.0))
        family_ax.barh(0, share, left=left, height=0.38, color=FAMILY_COLORS[name], edgecolor="white", linewidth=0.7)
        if share >= 0.065:
            family_ax.text(left + share / 2, 0, f"{share * 100:.1f}%", ha="center", va="center", color="white" if name in {"2D descriptors", "Three-conformer xTB"} else INK, fontsize=7.4, fontweight="bold")
        left += share
    family_ax.set_xlim(0, 1)
    family_ax.set_ylim(-0.42, 0.56)
    family_ax.axis("off")
    family_ax.text(0, 0.43, "Global |TreeSHAP| contribution by feature family", ha="left", va="bottom", color=INK, fontsize=8.4, fontweight="bold")

    handles = [mpl.patches.Patch(facecolor=FAMILY_COLORS[name], edgecolor="none", label=name) for name in family_order]
    family_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, -0.24), ncol=5, frameon=False, fontsize=6.9, handlelength=1.15, columnspacing=1.15, handletextpad=0.35, borderaxespad=0)

    cax = family_ax.inset_axes([0.79, 0.84, 0.19, 0.095])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(0, 1), orientation="horizontal")
    cax.set_xticks([0, 1], labels=["Low", "High"])
    cax.tick_params(axis="x", labelsize=6.7, length=0, pad=1.2)
    cax.set_title("Feature value", fontsize=6.9, color=MUTED, pad=1.5)
    for spine in cax.spines.values():
        spine.set_visible(False)

    rng = np.random.default_rng(20260719)
    for row, feature in enumerate(reversed(ordered)):
        subset = points.loc[points["feature"].eq(feature)].copy()
        x = subset["weighted_standardized_shap"].to_numpy(float)
        y = swarm_offsets(x, row, rng)
        color = subset["feature_percentile"].fillna(0.5).to_numpy(float)
        draw_order = rng.permutation(len(subset))
        ax.scatter(x[draw_order], y[draw_order], c=color[draw_order], cmap=cmap, vmin=0, vmax=1, s=5.2, alpha=0.82, edgecolors="none", rasterized=True, zorder=3)

    ax.axvline(0, color=INK, linewidth=0.9, alpha=0.8, zorder=1)
    ax.set_xlim(-0.68, 0.55)
    ax.set_ylim(-0.65, len(ordered) - 0.35)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([DISPLAY.get(feature, feature) for feature in reversed(ordered)])
    ax.set_xlabel("Weighted standardized TreeSHAP contribution to D score", labelpad=7)
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="x", colors=INK)
    ax.tick_params(axis="y", pad=15, colors=INK)

    family_by_feature = global_importance.set_index("feature")["family"].to_dict()
    for tick, feature in zip(ax.get_yticklabels(), reversed(ordered)):
        tick.set_color(INK)
        ax.scatter(-0.013, tick.get_position()[1], s=27, marker="s", color=FAMILY_COLORS[family_by_feature[feature]], edgecolors="none", clip_on=False, zorder=5, transform=ax.get_yaxis_transform())

    fig.suptitle("Molecular descriptors and dipole physics provide complementary signals", x=0.085, y=0.985, ha="left", va="top", fontsize=12.1, fontweight="bold", color="#111111")
    fig.subplots_adjust(left=0.285, right=0.985, top=0.91, bottom=0.105)

    for suffix in ("png", "pdf", "svg", "tiff"):
        kwargs = {"dpi": 600} if suffix == "tiff" else ({"dpi": 300} if suffix == "png" else {})
        fig.savefig(STEM.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)

    physics_share = 1.0 - float(family_share["2D descriptors"])
    manifest = {
        "surface": "paper_main",
        "core_conclusion": "The tabular component combines a distributed 2D descriptor signal with multi-conformer and atom-resolved dipole physics.",
        "scope": "six nonzero-weight tabular members; the equivariant vector branch is excluded from feature-level attribution",
        "cohort": "frozen chemistry-curated 4,000-record evaluation set",
        "n_unique_domain_index": 3998,
        "method": "TreeSHAP divided by member prediction SD and averaged with normalized final tabular fusion weights",
        "statistics": {
            "two_dimensional_descriptor_share": float(family_share["2D descriptors"]),
            "quantum_and_3d_feature_share": physics_share,
            "top_feature": ordered[0],
            "top_feature_importance_fraction": float(global_importance.iloc[0]["importance_fraction"]),
        },
        "abbreviation_key": ABBREVIATION_KEY,
        "exports": [str(STEM.with_suffix(f".{suffix}")) for suffix in ("png", "pdf", "svg", "tiff")],
        "source_data": [
            str(SOURCE / "feature_interpretability_top15_points.csv"),
            str(SOURCE / "feature_interpretability_global_importance.csv"),
            str(SOURCE / "feature_interpretability_family_importance.csv"),
            str(SOURCE / "feature_interpretability_member_diagnostics.csv"),
        ],
        "reviewer_risk": "Attributions explain only the tabular component and are standardized before cross-member aggregation because constituent target scales differ.",
    }
    (OUT / "Fig_interpretability_feature_treeshap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest["statistics"], flush=True)


if __name__ == "__main__":
    main()
