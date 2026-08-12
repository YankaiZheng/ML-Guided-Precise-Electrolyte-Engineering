#!/usr/bin/env python3
"""Build paper-facing matched-pair design rules and novelty-error figures."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from adjustText import adjust_text
import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMMPA
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "analysis_outputs/qme14s_training/domain65k"
RUNS = DOMAIN / "model_runs"
OUT = ROOT / "analysis_outputs/paper_figures_ml_workflow"
SOURCE = OUT / "source_data"
DEV = DOMAIN / "domain65k_cv_pool_complete_features.csv"
FROZEN_DEV = RUNS / "domain65k_d_algorithm_comparison_fold0/candidate100_clean_fold0_3000_features.npz"
EVALUATION = RUNS / "domain65k_d_candidate100_fold0_fusion_multiobjective/evaluation4000_predictions.csv"
NEAREST = SOURCE / "fig2_nearest_tanimoto.csv"
SEED = 20260719

COLORS = {
    "deep_blue": "#4f779e",
    "sky_blue": "#91b4d1",
    "green": "#b6ccb9",
    "mauve": "#c7b4be",
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
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_manifest(stem: str, payload: dict) -> None:
    (OUT / f"{stem}_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_candidate_clean_development() -> pd.DataFrame:
    with np.load(FROZEN_DEV) as data:
        frozen_ids = set(data["domain_index"].astype(int).tolist())
    frame = pd.read_csv(DEV, usecols=["domain_index", "canonical_smiles", "D", "P"], low_memory=False)
    frame = frame.loc[frame["domain_index"].astype(int).isin(frozen_ids)].copy()
    if len(frame) != 60641:
        raise RuntimeError(f"Expected 60,641 frozen development rows, found {len(frame):,}")
    frame = frame.drop_duplicates("canonical_smiles").reset_index(drop=True)
    if len(frame) != 60600:
        raise RuntimeError(f"Expected 60,600 unique development molecules, found {len(frame):,}")
    frame["D_rank_pct"] = rankdata(frame["D"].to_numpy(float), method="average") / len(frame)
    frame["P_rank_pct"] = rankdata(frame["P"].to_numpy(float), method="average") / len(frame)
    return frame


def fragment_heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1
    return sum(atom.GetAtomicNum() > 0 for atom in mol.GetAtoms())


def canonical_fragment(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compact_fragment(smiles: str) -> str:
    text = smiles.replace("[*:1]", "*").replace("[1*]", "*")
    replacements = {
        "*C": "–CH₃",
        "C*": "–CH₃",
        "*F": "–F",
        "F*": "–F",
        "*Cl": "–Cl",
        "Cl*": "–Cl",
        "*Br": "–Br",
        "Br*": "–Br",
        "*N": "–NH₂",
        "N*": "–NH₂",
        "*O": "–OH",
        "O*": "–OH",
        "*C#N": "–CN",
        "N#C*": "–CN",
    }
    if text in replacements:
        return replacements[text]
    return text if len(text) <= 15 else text[:13] + "…"


def bootstrap_median(values: np.ndarray, rng: np.random.Generator, repeats: int = 1200) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def build_matched_pair_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    selected_path = SOURCE / "mmp_design_selected_transformations.csv"
    all_path = SOURCE / "mmp_design_all_transformations.csv"
    recurrent_path = SOURCE / "mmp_design_recurrent_transformations.csv"
    summary_path = SOURCE / "mmp_design_fragmentation_summary.json"
    if selected_path.exists() and all_path.exists() and recurrent_path.exists() and summary_path.exists():
        return pd.read_csv(all_path), pd.read_csv(selected_path), json.loads(summary_path.read_text(encoding="utf-8"))

    frame = load_candidate_clean_development()
    by_core: dict[str, list[tuple[str, float, float, int]]] = defaultdict(list)
    invalid = 0
    retained_fragmentations = 0
    for row in frame.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.canonical_smiles))
        if mol is None:
            invalid += 1
            continue
        seen: set[tuple[str, str]] = set()
        fragments = rdMMPA.FragmentMol(mol, maxCuts=1, maxCutBonds=25, resultsAsMols=False)
        for _, split in fragments:
            parts = split.split(".")
            if len(parts) != 2:
                continue
            first, second = canonical_fragment(parts[0]), canonical_fragment(parts[1])
            if first is None or second is None:
                continue
            h_first, h_second = fragment_heavy_atoms(first), fragment_heavy_atoms(second)
            if h_first == h_second:
                continue
            if h_first > h_second:
                core, variable, h_core, h_variable = first, second, h_first, h_second
            else:
                core, variable, h_core, h_variable = second, first, h_second, h_first
            if h_core < 4 or not (1 <= h_variable <= 6) or h_variable / (h_core + h_variable) > 0.42:
                continue
            key = (core, variable)
            if key in seen:
                continue
            seen.add(key)
            by_core[core].append((variable, float(row.D_rank_pct), float(row.P_rank_pct), int(row.domain_index)))
            retained_fragmentations += 1

    transform_records: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"d": [], "p": [], "core": []}
    )
    skipped_promiscuous_cores = 0
    used_cores = 0
    for core, records in by_core.items():
        unique = {}
        for variable, d_rank, p_rank, domain_index in records:
            unique.setdefault(variable, (d_rank, p_rank, domain_index))
        items = sorted(unique.items())
        if len(items) < 2:
            continue
        if len(items) > 100:
            skipped_promiscuous_cores += 1
            continue
        used_cores += 1
        for (var_a, a), (var_b, b) in combinations(items, 2):
            d_delta = b[0] - a[0]
            p_delta = b[1] - a[1]
            bucket = transform_records[(var_a, var_b)]
            bucket["d"].append(d_delta)
            bucket["p"].append(p_delta)
            bucket["core"].append(core)

    rows = []
    for (var_a, var_b), values in transform_records.items():
        d_delta = np.asarray(values["d"], dtype=float)
        p_delta = np.asarray(values["p"], dtype=float)
        n_pairs = len(d_delta)
        n_scaffolds = len(set(values["core"]))
        if n_pairs < 10 or n_scaffolds < 8:
            continue
        utility = 0.5 * (d_delta + p_delta)
        orientation = 1.0 if np.median(utility) >= 0 else -1.0
        if orientation < 0:
            var_a, var_b = var_b, var_a
            d_delta, p_delta, utility = -d_delta, -p_delta, -utility
        median_d, median_p, median_u = np.median(d_delta), np.median(p_delta), np.median(utility)
        consistency = float(np.mean(utility > 0))
        score = float(max(min(median_d, median_p), 0.0) * np.sqrt(np.log1p(n_pairs)) * consistency)
        rows.append(
            {
                "fragment_from": var_a,
                "fragment_to": var_b,
                "label_from": compact_fragment(var_a),
                "label_to": compact_fragment(var_b),
                "n_pairs": n_pairs,
                "n_scaffolds": n_scaffolds,
                "median_delta_D_rank": median_d,
                "median_delta_P_rank": median_p,
                "median_delta_utility": median_u,
                "direction_consistency": consistency,
                "selection_score": score,
                "_d_values": d_delta,
                "_p_values": p_delta,
                "_u_values": utility,
            }
        )
    recurrent = pd.DataFrame(rows)
    private = {"_d_values", "_p_values", "_u_values"}
    recurrent_public = recurrent[[column for column in recurrent.columns if column not in private]].copy()
    recurrent_public.to_csv(recurrent_path, index=False)
    candidates = recurrent.loc[
        (recurrent["median_delta_D_rank"] > 0)
        & (recurrent["median_delta_P_rank"] > 0)
        & (recurrent["direction_consistency"] >= 0.60)
    ].sort_values(["selection_score", "n_scaffolds"], ascending=False)
    if len(candidates) < 8:
        raise RuntimeError(f"Only {len(candidates)} favorable recurrent transformations passed the predeclared filters")

    rng = np.random.default_rng(SEED)
    enriched = []
    for _, row in candidates.head(40).iterrows():
        record = row.to_dict()
        for key, values in (
            ("D", row["_d_values"]),
            ("P", row["_p_values"]),
            ("utility", row["_u_values"]),
        ):
            low, high = bootstrap_median(np.asarray(values), rng)
            record[f"ci95_low_delta_{key}"] = low
            record[f"ci95_high_delta_{key}"] = high
        enriched.append(record)
    ranked = pd.DataFrame(enriched)
    robust = ranked.loc[ranked["ci95_low_delta_utility"] > 0].copy()
    if len(robust) < 8:
        robust = ranked.copy()

    selected_rows = []
    selected_nodes: set[str] = set()
    for _, row in robust.iterrows():
        new_nodes = {row.fragment_from, row.fragment_to} - selected_nodes
        if len(selected_nodes) + len(new_nodes) > 13 and len(selected_rows) >= 6:
            continue
        selected_rows.append(row)
        selected_nodes.update([row.fragment_from, row.fragment_to])
        if len(selected_rows) == 8:
            break
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if len(selected) < 8:
        selected = robust.head(8).reset_index(drop=True)

    all_public = ranked[[column for column in ranked.columns if column not in private]].copy()
    selected_public = selected[[column for column in selected.columns if column not in private]].copy()
    all_public.to_csv(all_path, index=False)
    selected_public.to_csv(selected_path, index=False)
    summary = {
        "development_rows_frozen": 60641,
        "unique_molecules": len(frame),
        "invalid_smiles": invalid,
        "retained_single_cut_fragmentations": retained_fragmentations,
        "cores_with_pairs": used_cores,
        "promiscuous_cores_over_100_variants_excluded": skipped_promiscuous_cores,
        "recurrent_transformations_total": len(recurrent_public),
        "recurrent_favorable_transformations": len(candidates),
        "selected_transformations": len(selected_public),
        "selection_filters": {
            "minimum_pairs": 10,
            "minimum_scaffolds": 8,
            "minimum_direction_consistency": 0.60,
            "positive_median_D_and_P_rank_shift": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return all_public, selected_public, summary


def plot_matched_pair_design_rules(all_transformations: pd.DataFrame, selected: pd.DataFrame, summary: dict) -> None:
    selected = selected.copy()
    selected["transformation"] = selected["label_from"] + " → " + selected["label_to"]
    graph = nx.DiGraph()
    for row in selected.itertuples(index=False):
        graph.add_edge(row.label_from, row.label_to, n=row.n_pairs, utility=row.median_delta_utility)
    source_nodes = list(dict.fromkeys(selected["label_from"].tolist()))
    target_nodes = list(dict.fromkeys(selected["label_to"].tolist()))
    source_y = np.linspace(0.88, -0.88, len(source_nodes))
    target_y = np.linspace(1.28, -1.28, len(target_nodes))
    positions = {node: (0.0, y) for node, y in zip(source_nodes, source_y)}
    positions.update({node: (1.0, y) for node, y in zip(target_nodes, target_y)})

    fig = plt.figure(figsize=(7.2, 5.35))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.15], width_ratios=[0.98, 1.08], hspace=0.44, wspace=0.38)
    ax_network = fig.add_subplot(grid[0, 0])
    ax_cloud = fig.add_subplot(grid[0, 1])
    ax_ci = fig.add_subplot(grid[1, :])

    node_sizes = []
    for node in graph.nodes:
        strength = sum(graph[u][v]["n"] for u, v in graph.edges if u == node or v == node)
        node_sizes.append(165 + 5 * np.sqrt(strength))
    nx.draw_networkx_nodes(
        graph, positions, ax=ax_network, node_size=node_sizes, node_color=COLORS["near_white"],
        edgecolors=COLORS["deep_blue"], linewidths=0.9,
    )
    widths = [0.8 + 2.2 * np.sqrt(graph[u][v]["n"] / selected["n_pairs"].max()) for u, v in graph.edges]
    nx.draw_networkx_edges(
        graph, positions, ax=ax_network, width=widths, edge_color=COLORS["deep_blue"],
        alpha=0.72, arrows=True, arrowsize=9, connectionstyle="arc3,rad=0.04",
        min_source_margin=7, min_target_margin=8,
    )
    nx.draw_networkx_labels(graph, positions, ax=ax_network, font_size=5.0, font_color=COLORS["ink"])
    ax_network.set_title("Recurrent favorable fragment substitutions")
    ax_network.text(
        0.02, -0.04, "arrow width scales with matched-pair count", transform=ax_network.transAxes,
        fontsize=5.8, color=COLORS["mid_gray"], ha="left", va="bottom",
    )
    ax_network.margins(x=0.18, y=0.16)
    ax_network.set_axis_off()
    panel_label(ax_network, "a", x=-0.05, y=1.08)

    cloud = all_transformations.copy()
    x = 100 * cloud["median_delta_P_rank"].to_numpy(float)
    y = 100 * cloud["median_delta_D_rank"].to_numpy(float)
    sizes = 8 + 7 * np.sqrt(cloud["n_pairs"].to_numpy(float))
    ax_cloud.axhspan(0, max(1, y.max() * 1.12), color=COLORS["green"], alpha=0.13, zorder=0)
    ax_cloud.axvspan(0, max(1, x.max() * 1.12), color=COLORS["green"], alpha=0.13, zorder=0)
    ax_cloud.axhline(0, color=COLORS["light_gray"], linewidth=0.8)
    ax_cloud.axvline(0, color=COLORS["light_gray"], linewidth=0.8)
    scatter = ax_cloud.scatter(
        x, y, s=sizes, c=cloud["direction_consistency"], cmap=LinearSegmentedColormap.from_list(
            "consistency", [COLORS["light_gray"], COLORS["sky_blue"], COLORS["deep_blue"]]
        ), vmin=0.60, vmax=max(0.85, float(cloud["direction_consistency"].max())),
        edgecolor="white", linewidth=0.35, alpha=0.88, zorder=2,
    )
    for row in selected.itertuples(index=False):
        dx, dy = 100 * row.median_delta_P_rank, 100 * row.median_delta_D_rank
        ax_cloud.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": COLORS["peach"], "lw": 0.85, "alpha": 0.85},
            zorder=3,
        )
    cbar = fig.colorbar(scatter, ax=ax_cloud, fraction=0.048, pad=0.03)
    cbar.set_label("Direction consistency", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.8)
    ax_cloud.set(
        xlabel="Median ΔP percentile (points)", ylabel="Median ΔD percentile (points)",
        title="Matched-pair effect map",
    )
    ax_cloud.text(
        0.98, 0.04, "upper-right: both targets improve", transform=ax_cloud.transAxes,
        ha="right", va="bottom", fontsize=5.8, color="#5f7968",
    )
    panel_label(ax_cloud, "b", x=-0.17, y=1.08)

    ordered = selected.sort_values("median_delta_utility").reset_index(drop=True)
    ypos = np.arange(len(ordered))
    for idx, row in ordered.iterrows():
        ax_ci.plot(
            [100 * row.ci95_low_delta_D, 100 * row.ci95_high_delta_D], [idx + 0.12, idx + 0.12],
            color=COLORS["deep_blue"], linewidth=1.2,
        )
        ax_ci.scatter(100 * row.median_delta_D_rank, idx + 0.12, s=24, color=COLORS["deep_blue"], edgecolor="white", linewidth=0.45, zorder=3)
        ax_ci.plot(
            [100 * row.ci95_low_delta_P, 100 * row.ci95_high_delta_P], [idx - 0.12, idx - 0.12],
            color=COLORS["peach"], linewidth=1.2,
        )
        ax_ci.scatter(100 * row.median_delta_P_rank, idx - 0.12, s=24, color=COLORS["peach"], edgecolor="white", linewidth=0.45, zorder=3)
        ax_ci.text(
            max(100 * row.ci95_high_delta_D, 100 * row.ci95_high_delta_P) + 0.35, idx,
            f"n={int(row.n_pairs)}, {int(row.n_scaffolds)} cores", fontsize=5.6,
            color=COLORS["mid_gray"], va="center",
        )
    ax_ci.axvline(0, color=COLORS["mid_gray"], linewidth=0.75, linestyle="--")
    ax_ci.set_yticks(ypos, ordered["transformation"])
    ax_ci.set_xlabel("Median matched-pair rank shift (percentile points; bootstrap 95% CI)")
    ax_ci.set_title("Cross-scaffold effect estimates")
    ax_ci.legend(
        handles=[
            Line2D([0], [0], marker="o", color=COLORS["deep_blue"], label="D rank", markersize=4),
            Line2D([0], [0], marker="o", color=COLORS["peach"], label="P rank", markersize=4),
        ], loc="lower right",
    )
    panel_label(ax_ci, "c", x=-0.075, y=1.08)

    save_figure(fig, "Fig_design_matched_pair_atlas")
    write_manifest(
        "Fig_design_matched_pair_atlas",
        {
            "surface": "paper_main",
            "core_conclusion": "Recurrent local fragment substitutions are associated with concordant upward shifts in D and P ranks across multiple molecular cores.",
            "archetype": "asymmetric mixed-modality figure",
            "backend": "Python/matplotlib and RDKit MMPA",
            "development_scope": summary,
            "statistics": "Median paired percentile-rank shifts with nonparametric bootstrap 95% confidence intervals.",
            "selection_note": "Favorable direction is oriented after grouping each unordered fragment transformation; associations are not causal effects.",
            "source_data": [
                str(SOURCE / "mmp_design_all_transformations.csv"),
                str(SOURCE / "mmp_design_selected_transformations.csv"),
                str(SOURCE / "mmp_design_fragmentation_summary.json"),
            ],
            "reviewer_risk": "Transformations are discovered and summarized within the development set; claims must remain associative and report pair/core counts.",
        },
    )


def plot_matched_pair_effect_map(
    recurrent: pd.DataFrame, selected: pd.DataFrame, summary: dict
) -> None:
    recurrent = recurrent.copy()
    selected = selected.copy().reset_index(drop=True)
    selected["rule_id"] = [f"R{i}" for i in range(1, len(selected) + 1)]
    selected["transformation"] = selected["label_from"] + " → " + selected["label_to"]

    x = 100 * recurrent["median_delta_P_rank"].to_numpy(float)
    y = 100 * recurrent["median_delta_D_rank"].to_numpy(float)
    cap = float(np.quantile(recurrent["n_scaffolds"], 0.98))
    sizes = 5.0 + 2.4 * np.sqrt(np.minimum(recurrent["n_scaffolds"].to_numpy(float), cap))
    consistency_cmap = LinearSegmentedColormap.from_list(
        "mmp_consistency", [COLORS["light_gray"], COLORS["sky_blue"], COLORS["deep_blue"]]
    )
    norm = Normalize(vmin=0.50, vmax=1.00)
    extent = max(80.0, 10.0 * np.ceil(max(np.max(np.abs(x)), np.max(np.abs(y))) / 10.0))
    extent = min(100.0, extent)

    fig = plt.figure(figsize=(7.2, 4.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.62, 0.88], wspace=0.19)
    ax = fig.add_subplot(grid[0, 0])
    ax_rules = fig.add_subplot(grid[0, 1])

    ax.axvspan(0, extent, ymin=0.5, ymax=1.0, color=COLORS["green"], alpha=0.18, zorder=0)
    ax.axhline(0, color=COLORS["mid_gray"], linewidth=0.75, linestyle="--", zorder=1)
    ax.axvline(0, color=COLORS["mid_gray"], linewidth=0.75, linestyle="--", zorder=1)
    cloud = ax.scatter(
        x, y, s=sizes, c=recurrent["direction_consistency"], cmap=consistency_cmap,
        norm=norm, alpha=0.48, linewidth=0, rasterized=True, zorder=2,
    )

    selected_x = 100 * selected["median_delta_P_rank"].to_numpy(float)
    selected_y = 100 * selected["median_delta_D_rank"].to_numpy(float)
    selected_sizes = 34 + 3.0 * np.sqrt(selected["n_scaffolds"].to_numpy(float))
    ax.scatter(
        selected_x, selected_y, s=selected_sizes, color=COLORS["deep_blue"],
        edgecolor=COLORS["peach"], linewidth=1.5, zorder=4,
    )
    texts = []
    for row in selected.itertuples(index=False):
        texts.append(
            ax.text(
                100 * row.median_delta_P_rank, 100 * row.median_delta_D_rank, row.rule_id,
                fontsize=6.2, fontweight="bold", color=COLORS["ink"], zorder=5,
            )
        )
    adjust_text(
        texts, ax=ax, expand=(1.35, 1.45), force_text=(0.45, 0.55),
        arrowprops={"arrowstyle": "-", "color": COLORS["peach"], "lw": 0.65},
    )

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Median ΔP rank (percentile points)")
    ax.set_ylabel("Median ΔD rank (percentile points)")
    ax.set_title("Matched-pair substitution landscape", loc="left", pad=20)
    ax.text(
        0.0, 1.035,
        "Each point: one recurrent replacement (≥10 pairs; ≥8 scaffolds); direction is oriented toward higher mean D/P rank",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=5.45, color=COLORS["mid_gray"],
    )
    ax.text(
        0.97, 0.97, "joint gain\nin D and P", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.0, color="#5f7968", fontweight="bold",
    )
    ax.text(
        0.03, 0.97, "D gain\nP loss", transform=ax.transAxes,
        ha="left", va="top", fontsize=5.7, color=COLORS["mid_gray"],
    )
    ax.text(
        0.97, 0.23, "P gain\nD loss", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=5.7, color=COLORS["mid_gray"],
    )
    ax.grid(color=COLORS["light_gray"], linewidth=0.42, alpha=0.55, zorder=0)

    cbar = fig.colorbar(cloud, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Direction consistency", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.6)
    size_handles = []
    for n in (10, 50, 250):
        marker_size = np.sqrt(5.0 + 2.4 * np.sqrt(min(n, cap)))
        size_handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["sky_blue"],
                   markeredgecolor="none", markersize=marker_size, label=f"{n} cores")
        )
    selected_handle = Line2D(
        [0], [0], marker="o", linestyle="", markerfacecolor=COLORS["deep_blue"],
        markeredgecolor=COLORS["peach"], markeredgewidth=1.2, markersize=5.5,
        label="Highlighted joint-gain rule",
    )
    ax.legend(
        handles=[*size_handles, selected_handle], title="Support / selection",
        loc="lower left", bbox_to_anchor=(0.015, 0.015), fontsize=5.3, title_fontsize=5.7,
    )

    ax_rules.set_axis_off()
    ax_rules.text(0.0, 0.98, "Highlighted rules", fontsize=7.8, fontweight="bold", va="top")
    ax_rules.text(
        0.0, 0.92, "Abbreviated substituent notation", fontsize=5.6,
        color=COLORS["mid_gray"], va="top",
    )
    short_rules = [
        r"–F  $→$  –SOCH$_3$",
        r"–F  $→$  –SO$_2$CH$_3$",
        r"–F  $→$  –SO$_2$NH$_2$",
        r"–OCH$_3$  $→$  –SOCH$_3$",
        r"–OH  $→$  –SO$_3$H",
        r"–OCH$_3$  $→$  –SO$_2$CH$_3$",
        r"–F  $→$  –SCN",
        r"–OH  $→$  –CH$_2$CN",
    ]
    y_positions = np.linspace(0.80, 0.27, len(short_rules))
    for idx, (row, label) in enumerate(zip(selected.itertuples(index=False), short_rules)):
        xpos, ypos = 0.0, y_positions[idx]
        ax_rules.text(
            xpos, ypos, row.rule_id, fontsize=6.8, fontweight="bold",
            color=COLORS["deep_blue"], va="center",
        )
        ax_rules.text(
            xpos + 0.13, ypos, label, fontsize=6.4,
            color=COLORS["ink"], va="center",
        )
    ax_rules.text(
        0.0, 0.10,
        "Point position: median rank shifts\nPoint size: independent scaffold support\nColor: directional consistency",
        fontsize=5.45, color=COLORS["mid_gray"], va="bottom", linespacing=1.45,
    )
    ax_rules.text(
        0.0, 0.0,
        f"{summary['recurrent_transformations_total']:,} recurrent transformations; "
        f"{summary['unique_molecules']:,} molecules",
        fontsize=5.15, color=COLORS["mid_gray"], va="bottom",
    )
    save_figure(fig, "Fig_design_matched_pair_effect_map")
    write_manifest(
        "Fig_design_matched_pair_effect_map",
        {
            "surface": "paper_main",
            "core_conclusion": "Recurrent local substitutions define a broad D-P trade-off landscape, with eight cross-scaffold rules showing concordant joint rank gains.",
            "archetype": "annotated effect landscape with rule key",
            "backend": "Python/matplotlib, RDKit MMPA, and adjustText",
            "development_scope": summary,
            "statistics": "Median paired percentile-rank shifts; highlighted rules require a positive bootstrap 95% interval for joint utility.",
            "orientation_note": "Each unordered substitution is oriented toward nonnegative median joint utility before plotting; mixed-sign points represent D-P trade-offs.",
            "source_data": [
                str(SOURCE / "mmp_design_recurrent_transformations.csv"),
                str(SOURCE / "mmp_design_selected_transformations.csv"),
                str(SOURCE / "mmp_design_fragmentation_summary.json"),
            ],
            "reviewer_risk": "This is development-set matched-pair association, not a causal intervention analysis.",
        },
    )


def build_dmtmsa_motif_axes() -> tuple[pd.DataFrame, dict]:
    frame = load_candidate_clean_development()
    motif_specs = [
        ("SO2NMe2", "S(=O)(=O)N(C)C", r"SO$_2$NMe$_2$"),
        ("CF3SO2", "S(=O)(=O)C(F)(F)F", r"CF$_3$SO$_2$"),
    ]
    rows = []
    masks = {}
    for motif, smarts, display in motif_specs:
        query = Chem.MolFromSmarts(smarts)
        mask = frame["canonical_smiles"].map(
            lambda smiles: bool(Chem.MolFromSmiles(smiles).HasSubstructMatch(query))
        )
        masks[motif] = mask
        subset = frame.loc[mask]
        rows.append(
            {
                "motif": motif,
                "display": display,
                "smarts": smarts,
                "n_molecules": int(len(subset)),
                "median_D_rank_pct": float(100 * subset["D_rank_pct"].median()),
                "median_P_rank_pct": float(100 * subset["P_rank_pct"].median()),
            }
        )
    axes = pd.DataFrame(rows)
    exact_intersection = int((masks["SO2NMe2"] & masks["CF3SO2"]).sum())
    axes.to_csv(SOURCE / "dmtmsa_motif_axes.csv", index=False)
    return axes, {"exact_motif_intersection_in_development": exact_intersection}


def plot_matched_pair_effect_panel(recurrent: pd.DataFrame, motifs: pd.DataFrame, motif_summary: dict) -> None:
    recurrent = recurrent.copy()

    x = 100 * recurrent["median_delta_P_rank"].to_numpy(float)
    y = 100 * recurrent["median_delta_D_rank"].to_numpy(float)
    cap = float(np.quantile(recurrent["n_scaffolds"], 0.98))
    sizes = 4.0 + 2.0 * np.sqrt(np.minimum(recurrent["n_scaffolds"].to_numpy(float), cap))
    cmap = LinearSegmentedColormap.from_list(
        "mmp_consistency_panel", [COLORS["light_gray"], COLORS["sky_blue"], COLORS["deep_blue"]]
    )
    norm = Normalize(vmin=0.50, vmax=1.00)

    fig, ax = plt.subplots(figsize=(3.55, 3.18))
    ax.axvspan(0, 90, ymin=0.5, ymax=1.0, color=COLORS["green"], alpha=0.17, zorder=0)
    ax.axhline(0, color=COLORS["mid_gray"], linewidth=0.7, linestyle="--", zorder=1)
    ax.axvline(0, color=COLORS["mid_gray"], linewidth=0.7, linestyle="--", zorder=1)
    cloud = ax.scatter(
        x, y, s=sizes, c=recurrent["direction_consistency"], cmap=cmap, norm=norm,
        alpha=0.46, linewidth=0, rasterized=True, zorder=2,
    )

    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Median ΔP rank (pp)")
    ax.set_ylabel("Median ΔD rank (pp)")
    ax.set_title("Matched-pair design landscape")
    ax.grid(color=COLORS["light_gray"], linewidth=0.40, alpha=0.55, zorder=0)

    # Place the two DMTMSA motif families by their observed median rank enrichment
    # relative to the 50th-percentile development-set baseline.
    motif_x = motifs["median_P_rank_pct"].to_numpy(float) - 50.0
    motif_y = motifs["median_D_rank_pct"].to_numpy(float) - 50.0
    ax.scatter(
        motif_x, motif_y, s=[135, 115], color=COLORS["deep_blue"],
        edgecolor=COLORS["peach"], linewidth=1.35, zorder=4,
    )
    motif_offsets = [(4, 7, "left"), (4, -9, "left")]
    for row, mx, my, (dx, dy, align) in zip(motifs.itertuples(index=False), motif_x, motif_y, motif_offsets):
        ax.annotate(
            row.display, (mx, my), xytext=(dx, dy), textcoords="offset points",
            ha=align, va="center", fontsize=5.7, fontweight="bold", color=COLORS["ink"], zorder=5,
        )

    cbar = fig.colorbar(cloud, ax=ax, fraction=0.040, pad=0.025)
    cbar.set_label("Consistency", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.6)
    handles = []
    for n in (10, 50, 250):
        marker_size = np.sqrt(4.0 + 2.0 * np.sqrt(min(n, cap)))
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="", markerfacecolor=COLORS["sky_blue"],
                markeredgecolor="none", markersize=marker_size, label=str(n),
            )
        )
    ax.legend(
        handles=handles, title="n cores", loc="lower left", bbox_to_anchor=(0.01, 0.01),
        fontsize=5.2, title_fontsize=5.5, handletextpad=0.5, labelspacing=0.35,
    )

    save_figure(fig, "Fig_design_matched_pair_effect_map_panel")
    write_manifest(
        "Fig_design_matched_pair_effect_map_panel",
        {
            "surface": "paper_main_subpanel",
            "core_conclusion": "The two DMTMSA-relevant motif families occupy the joint-positive region when plotted by their median D/P rank enrichment relative to the development-set median.",
            "motif_summary": motif_summary,
            "source_data": [
                str(SOURCE / "mmp_design_recurrent_transformations.csv"),
                str(SOURCE / "dmtmsa_motif_axes.csv"),
            ],
            "caption_required": "Define MMP point, color, size, orientation, and specify that the two peach-outlined motif bubbles are family median rank enrichments relative to the 50th-percentile development-set baseline.",
        },
    )


def build_novelty_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    evaluation = pd.read_csv(EVALUATION)
    similarity = pd.read_csv(NEAREST)
    similarity = similarity.loc[similarity["group"].eq("Evaluation")].reset_index(drop=True)
    if len(evaluation) != 4000 or len(similarity) != 4000:
        raise RuntimeError("Novelty analysis requires exactly 4,000 evaluation and similarity rows")
    if not np.array_equal(evaluation["domain_index"].to_numpy(int), similarity["id"].to_numpy(int)):
        raise RuntimeError("Evaluation similarity rows are not aligned with frozen evaluation predictions")
    frame = evaluation[["locked_row_id", "domain_index", "canonical_smiles", "D", "score__final_fusion_multiobjective"]].copy()
    frame["nearest_tanimoto"] = similarity["nearest_tanimoto"].to_numpy(float)
    frame["D_rank_pct"] = rankdata(frame["D"].to_numpy(float), method="average") / len(frame)
    frame["pred_rank_pct"] = rankdata(frame["score__final_fusion_multiobjective"].to_numpy(float), method="average") / len(frame)
    frame["abs_rank_error_pp"] = 100 * np.abs(frame["D_rank_pct"] - frame["pred_rank_pct"])
    similarity_edges = [0.0, 0.45, 0.55, 0.65, 0.75, 1.000001]
    similarity_labels = ["<0.45", "0.45–0.55", "0.55–0.65", "0.65–0.75", "≥0.75"]
    frame["similarity_bin"] = pd.cut(
        frame["nearest_tanimoto"], bins=similarity_edges, labels=similarity_labels,
        include_lowest=True, right=False,
    )
    frame["D_quintile"] = pd.qcut(
        frame["D"], 5, labels=["Q1\nlowest D", "Q2", "Q3", "Q4", "Q5\nhighest D"]
    )
    cells = (
        frame.groupby(["D_quintile", "similarity_bin"], observed=False)
        .agg(
            n=("abs_rank_error_pp", "size"),
            median_abs_rank_error_pp=("abs_rank_error_pp", "median"),
            mean_similarity=("nearest_tanimoto", "mean"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(SEED + 1)
    profile_rows = []
    for label in similarity_labels:
        subset = frame.loc[frame["similarity_bin"].astype(str).eq(label)].copy()
        low, high = bootstrap_median(subset["abs_rank_error_pp"].to_numpy(float), rng)
        profile_rows.append(
            {
                "similarity_bin": label,
                "n": len(subset),
                "median_similarity": subset["nearest_tanimoto"].median(),
                "median_abs_rank_error_pp": subset["abs_rank_error_pp"].median(),
                "ci95_low_median_error": low,
                "ci95_high_median_error": high,
                "spearman_within_bin": spearmanr(
                    subset["D"], subset["score__final_fusion_multiobjective"]
                ).statistic,
            }
        )
    profile = pd.DataFrame(profile_rows)
    rho = float(spearmanr(frame["nearest_tanimoto"], frame["abs_rank_error_pp"]).statistic)
    summary = {
        "n_evaluation": len(frame),
        "n_unique_canonical_smiles": int(frame["canonical_smiles"].nunique()),
        "similarity_range": [float(frame["nearest_tanimoto"].min()), float(frame["nearest_tanimoto"].max())],
        "spearman_similarity_vs_abs_rank_error": rho,
        "error_definition": "Absolute difference between frozen-fusion and reference D percentile ranks, in percentile points.",
        "similarity_definition": "Nearest Morgan radius-2 Tanimoto similarity to candidate100-clean development molecules.",
    }
    frame.to_csv(SOURCE / "novelty_error_phase_points.csv", index=False)
    cells.to_csv(SOURCE / "novelty_error_phase_cells.csv", index=False)
    profile.to_csv(SOURCE / "novelty_error_profile.csv", index=False)
    return frame, cells, profile, summary


def plot_novelty_error_phase(cells: pd.DataFrame, profile: pd.DataFrame, summary: dict) -> None:
    similarity_labels = ["<0.45", "0.45–0.55", "0.55–0.65", "0.65–0.75", "≥0.75"]
    d_labels = ["Q1\nlowest D", "Q2", "Q3", "Q4", "Q5\nhighest D"]
    matrix = cells.pivot(index="D_quintile", columns="similarity_bin", values="median_abs_rank_error_pp").reindex(index=d_labels, columns=similarity_labels)
    counts = cells.pivot(index="D_quintile", columns="similarity_bin", values="n").reindex(index=d_labels, columns=similarity_labels)
    values = matrix.to_numpy(float)
    cmap = LinearSegmentedColormap.from_list(
        "error_blue", [COLORS["near_white"], "#d8e3ea", COLORS["sky_blue"], COLORS["deep_blue"]]
    )
    vmax = float(np.nanquantile(values, 0.95))
    norm = Normalize(vmin=max(0.0, float(np.nanmin(values)) * 0.90), vmax=vmax)

    fig = plt.figure(figsize=(7.2, 3.55))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.42, 0.90], wspace=0.50)
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_profile = fig.add_subplot(grid[0, 1])
    image = ax_heat.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax_heat.set_xticks(np.arange(len(similarity_labels)), similarity_labels)
    ax_heat.set_yticks(np.arange(len(d_labels)), d_labels)
    ax_heat.set_xlabel("Nearest training-set Tanimoto similarity")
    ax_heat.set_ylabel("Reference D quintile")
    ax_heat.set_title("Absolute D rank error across chemical space")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            n = int(counts.iloc[row, col])
            color = "white" if norm(value) > 0.62 else COLORS["ink"]
            ax_heat.text(col, row + 0.08, f"{value:.1f}", ha="center", va="center", fontsize=7.0, fontweight="bold", color=color)
            ax_heat.text(col, row - 0.19, f"n={n}", ha="center", va="center", fontsize=5.3, color=color, alpha=0.88)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.035)
    cbar.ax.set_title("Error\n(pp)", fontsize=5.8, pad=4)
    cbar.ax.tick_params(labelsize=5.8)
    panel_label(ax_heat, "a", x=-0.14, y=1.08)

    x = profile["median_similarity"].to_numpy(float)
    y = profile["median_abs_rank_error_pp"].to_numpy(float)
    yerr = np.vstack(
        [y - profile["ci95_low_median_error"].to_numpy(float), profile["ci95_high_median_error"].to_numpy(float) - y]
    )
    ax_profile.plot(x, y, color=COLORS["deep_blue"], linewidth=1.5, zorder=2)
    ax_profile.errorbar(
        x, y, yerr=yerr, fmt="o", markersize=5.5, color=COLORS["deep_blue"],
        ecolor=COLORS["sky_blue"], elinewidth=1.1, capsize=2.5, markeredgecolor="white", markeredgewidth=0.6,
        zorder=3,
    )
    label_offsets = [(-6, 8), (0, 9), (-10, 9), (10, 7), (0, 9)]
    for row, offset in zip(profile.itertuples(index=False), label_offsets):
        ax_profile.annotate(
            f"n={row.n}", (row.median_similarity, row.median_abs_rank_error_pp),
            xytext=offset, textcoords="offset points", ha="center", fontsize=5.5, color=COLORS["mid_gray"],
        )
    ax_profile.set(
        xlabel="Median similarity within bin", ylabel="Median absolute rank error\n(percentile points)",
        title="Novelty–error profile",
    )
    ax_profile.text(
        0.04, 0.05,
        f"Spearman ρ = {summary['spearman_similarity_vs_abs_rank_error']:.2f}\n95% CI: bootstrap median error",
        transform=ax_profile.transAxes, ha="left", va="bottom", fontsize=6.0, color=COLORS["mid_gray"],
    )
    ax_profile.grid(axis="y", color=COLORS["light_gray"], linewidth=0.5, alpha=0.65)
    panel_label(ax_profile, "b", x=-0.20, y=1.08)

    save_figure(fig, "Fig_generalization_novelty_error_phase")
    write_manifest(
        "Fig_generalization_novelty_error_phase",
        {
            "surface": "paper_main",
            "core_conclusion": "Frozen-fusion ranking error is mapped jointly across chemical novelty and the reference D regime.",
            "archetype": "quantitative grid with marginal validation",
            "backend": "Python/matplotlib",
            "evaluation_scope": summary,
            "statistics": "Cell medians and bin-wise bootstrap 95% confidence intervals for median absolute percentile-rank error.",
            "source_data": [
                str(SOURCE / "novelty_error_phase_points.csv"),
                str(SOURCE / "novelty_error_phase_cells.csv"),
                str(SOURCE / "novelty_error_profile.csv"),
            ],
            "reviewer_risk": "This is a post-hoc applicability-domain analysis of frozen predictions; similarity was not used to retune the model.",
        },
    )


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    all_transformations, selected, mmp_summary = build_matched_pair_data()
    plot_matched_pair_design_rules(all_transformations, selected, mmp_summary)
    recurrent = pd.read_csv(SOURCE / "mmp_design_recurrent_transformations.csv")
    plot_matched_pair_effect_map(recurrent, selected, mmp_summary)
    motifs, motif_summary = build_dmtmsa_motif_axes()
    plot_matched_pair_effect_panel(recurrent, motifs, motif_summary)
    _, cells, profile, novelty_summary = build_novelty_data()
    plot_novelty_error_phase(cells, profile, novelty_summary)
    print(json.dumps({"mmp": mmp_summary, "novelty": novelty_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
