#!/usr/bin/env python3
"""Plot a label-free multiscale feature atlas for the Domain65k dataset.

The development set alone determines fingerprint projection, chemical clusters,
feature scaling, row order, and representative structures. The frozen 4,000-row
evaluation cohort and 78 candidates are only projected into that fixed atlas.
No D or P labels are loaded by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "analysis_outputs/qme14s_training/domain65k"
RUNS = DOMAIN / "model_runs"
LOCKED = RUNS / "domain65k_candidate100_final_locked"
OUT = ROOT / "analysis_outputs/paper_figures_ml_workflow"
SOURCE = OUT / "source_data"
CACHE = SOURCE / "feature_atlas_cache"

DEV_CSV = DOMAIN / "domain65k_cv_pool_complete_features.csv"
CURRENT4000 = LOCKED / "candidate100_clean_locked_dft_suitability_4194_random_remove194_seed20260715_kept4000_predictions.csv"
CANDIDATES = ROOT / "analysis_outputs/final_candidates_20260718/final78_source_order_predictions.csv"
EXCLUSION = ROOT / "analysis_outputs/candidate_curation_v3/domain65k_candidate100_exact_exclusion.csv"
FEATURE_DIR = DOMAIN / "features"
CHEAP = FEATURE_DIR / "domain65k_cheap_v3_v4_v5_features.pkl"
SINGLE = FEATURE_DIR / "domain65k_xtb_single_features.pkl"
MC3 = FEATURE_DIR / "domain65k_xtb_mc3_features.pkl"

RDKIT_PYTHON = Path("/opt/anaconda3/bin/python")
SEED = 20260722
N_CLUSTERS = 16
FP_SIZE = 1024

COLORS = {
    "deep_blue": "#4f779e",
    "sky_blue": "#91b4d1",
    "green": "#b6ccb9",
    "gray_green": "#aebcb5",
    "mauve": "#c7b4be",
    "peach": "#e1aca6",
    "ink": "#263238",
    "mid_gray": "#7b8790",
    "light_gray": "#dce3e7",
    "near_white": "#f7f9fa",
}

FEATURES = [
    # source, raw name, short display name, family
    ("cheap", "v3__mol_wt_calc", "MW", "Composition"),
    ("cheap", "v3__heavy_atoms_calc", "Heavy atoms", "Composition"),
    ("cheap", "v3__count_O", "O count", "Composition"),
    ("cheap", "v3__count_N", "N count", "Composition"),
    ("cheap", "v3__count_F", "F count", "Composition"),
    ("cheap", "v3__count_S", "S count", "Composition"),
    ("cheap", "v3__tpsa", "TPSA", "Topology"),
    ("cheap", "v3__mol_logp", "logP", "Topology"),
    ("cheap", "v3__rotatable_bonds_calc", "Rot. bonds", "Topology"),
    ("cheap", "v3__ring_count_calc", "Rings", "Topology"),
    ("cheap", "v3__aromatic_rings", "Arom. rings", "Topology"),
    ("cheap", "v3__fraction_csp3", "Csp3 frac.", "Topology"),
    ("cheap", "v3__gast_std", "Gasteiger σq", "Charge & shape"),
    ("cheap", "v3__gast_range", "Gasteiger Δq", "Charge & shape"),
    ("cheap", "v3__geom_radius_gyration", "Radius gyr.", "Charge & shape"),
    ("cheap", "v3__geom_asphericity", "Asphericity", "Charge & shape"),
    ("single", "xtb_full_dipole_debye", "xTB μ", "Single-conf. xTB"),
    ("single", "xtb_q_std", "xTB σq", "Single-conf. xTB"),
    ("single", "xtb_mol_polarizability_au", "xTB α", "Single-conf. xTB"),
    ("single", "xtb_quad_anisotropy", "Quad. aniso.", "Single-conf. xTB"),
    ("mc3", "xtb_mc3_xtb_full_dipole_debye_mean", "Mean μ (3 conf.)", "Three-conf. xTB"),
    ("mc3", "xtb_mc3_xtb_full_dipole_debye_std", "SD μ (3 conf.)", "Three-conf. xTB"),
    ("mc3", "xtb_mc3_xtb_full_dipole_debye_range", "Range μ (3 conf.)", "Three-conf. xTB"),
    ("mc3", "xtb_mc3_xtb_q_std_mean", "Mean σq (3 conf.)", "Three-conf. xTB"),
    ("mc3", "xtb_mc3_xtb_mol_polarizability_au_mean", "Mean α (3 conf.)", "Three-conf. xTB"),
]

FAMILY_COLORS = {
    "Composition": "#86a3bc",
    "Topology": COLORS["green"],
    "Charge & shape": COLORS["mauve"],
    "Single-conf. xTB": COLORS["peach"],
    "Three-conf. xTB": COLORS["deep_blue"],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rdkit_fingerprint_stage(input_csv: Path, output_npz: Path) -> None:
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
    matrix = np.zeros((len(rows), FP_SIZE), dtype=np.uint8)
    invalid: list[int] = []
    for index, row in enumerate(rows):
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            invalid.append(index)
            continue
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    if invalid:
        raise RuntimeError(f"Invalid SMILES rows in feature atlas input: {invalid[:20]}")
    np.savez_compressed(
        output_npz,
        fingerprints=matrix,
        input_sha256=np.asarray(file_sha256(input_csv)),
    )


def rdkit_render_stage(input_csv: Path, output_dir: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    output_dir.mkdir(parents=True, exist_ok=True)
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise RuntimeError(f"Cannot render representative SMILES: {row['canonical_smiles']}")
        drawer = rdMolDraw2D.MolDraw2DCairo(360, 150)
        options = drawer.drawOptions()
        options.clearBackground = False
        options.padding = 0.08
        options.bondLineWidth = 1.8
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        (output_dir / f"{row['cluster_id']}.png").write_bytes(drawer.GetDrawingText())


def configure_style(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def ensure_rdkit_cache(smiles_csv: Path, fingerprint_npz: Path) -> None:
    import numpy as np

    expected_hash = file_sha256(smiles_csv)
    valid = False
    if fingerprint_npz.exists():
        cached = np.load(fingerprint_npz, allow_pickle=False)
        valid = str(cached["input_sha256"].item()) == expected_hash
    if valid:
        return
    if not RDKIT_PYTHON.exists():
        raise FileNotFoundError(f"RDKit runtime not found: {RDKIT_PYTHON}")
    subprocess.run(
        [str(RDKIT_PYTHON), str(Path(__file__).resolve()), "--rdkit-fingerprints", str(smiles_csv), str(fingerprint_npz)],
        check=True,
    )


def load_feature_matrix(dev_ids, pd):
    import numpy as np

    pieces = []
    paths = {"cheap": CHEAP, "single": SINGLE, "mc3": MC3}
    for source, path in paths.items():
        columns = [raw for src, raw, _, _ in FEATURES if src == source]
        frame = pd.read_pickle(path)[["domain_index", *columns]].set_index("domain_index")
        aligned = frame.reindex(dev_ids)
        if aligned.index.has_duplicates or aligned[columns].isna().all(axis=1).any():
            raise RuntimeError(f"Feature alignment failed for {source}")
        pieces.append(aligned[columns].reset_index(drop=True))
    raw = pd.concat(pieces, axis=1)
    ordered_raw = [raw_name for _, raw_name, _, _ in FEATURES]
    raw = raw[ordered_raw].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    if raw.isna().any().any():
        raw = raw.fillna(raw.median(numeric_only=True))
    raw.columns = [short_name for _, _, short_name, _ in FEATURES]
    return raw


def main_plot() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import leaves_list, linkage
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD

    configure_style(mpl)
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(DEV_CSV, usecols=["domain_index", "canonical_smiles"], low_memory=False)
    excluded = set(pd.read_csv(EXCLUSION, usecols=["domain_index"])["domain_index"].astype(int))
    dev = dev.loc[~dev["domain_index"].astype(int).isin(excluded)].reset_index(drop=True)
    evaluation = pd.read_csv(CURRENT4000, usecols=["domain_index", "canonical_smiles"], low_memory=False)
    candidates = pd.read_csv(CANDIDATES, usecols=["candidate_id", "canonical_smiles"], low_memory=False)
    if (len(dev), len(evaluation), len(candidates)) != (60641, 4000, 78):
        raise RuntimeError(f"Unexpected cohort sizes: development={len(dev)}, evaluation={len(evaluation)}, candidates={len(candidates)}")

    atlas_input = pd.concat(
        [
            dev.assign(group="Development", row_id=dev["domain_index"].astype(str)),
            evaluation.assign(group="Evaluation", row_id=evaluation["domain_index"].astype(str)),
            candidates.assign(group="Candidates", row_id=candidates["candidate_id"].astype(str)),
        ],
        ignore_index=True,
    )[["group", "row_id", "canonical_smiles"]]
    smiles_csv = CACHE / "feature_atlas_smiles.csv"
    atlas_input.to_csv(smiles_csv, index=False)
    fingerprint_npz = CACHE / "feature_atlas_morgan1024.npz"
    ensure_rdkit_cache(smiles_csv, fingerprint_npz)
    fingerprints = np.load(fingerprint_npz, allow_pickle=False)["fingerprints"]
    if fingerprints.shape != (len(atlas_input), FP_SIZE):
        raise RuntimeError(f"Unexpected fingerprint matrix shape: {fingerprints.shape}")

    n_dev = len(dev)
    n_eval = len(evaluation)
    fp_dev = fingerprints[:n_dev].astype(np.float32, copy=False)
    fp_eval = fingerprints[n_dev:n_dev + n_eval].astype(np.float32, copy=False)
    fp_candidates = fingerprints[n_dev + n_eval:].astype(np.float32, copy=False)

    reducer = TruncatedSVD(n_components=32, n_iter=8, random_state=SEED)
    latent_dev = reducer.fit_transform(fp_dev)
    latent_eval = reducer.transform(fp_eval)
    latent_candidates = reducer.transform(fp_candidates)
    clusterer = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=SEED,
        batch_size=2048,
        n_init=20,
        max_iter=300,
        reassignment_ratio=0.005,
    )
    cluster_dev = clusterer.fit_predict(latent_dev)
    cluster_eval = clusterer.predict(latent_eval)
    cluster_candidates = clusterer.predict(latent_candidates)

    feature_values = load_feature_matrix(dev["domain_index"].astype(int).to_numpy(), pd)
    center = feature_values.median(axis=0)
    iqr = feature_values.quantile(0.75) - feature_values.quantile(0.25)
    robust_scale = (iqr / 1.349).replace(0, 1.0)
    standardized = ((feature_values - center) / robust_scale).clip(-4.0, 4.0)
    cluster_medians = standardized.assign(cluster=cluster_dev).groupby("cluster", sort=True).median()
    heat_for_order = cluster_medians.to_numpy(float)
    row_linkage = linkage(heat_for_order, method="average", metric="euclidean")
    old_order = leaves_list(row_linkage).astype(int).tolist()
    old_to_new = {old: new for new, old in enumerate(old_order)}
    cluster_dev_new = np.asarray([old_to_new[int(value)] for value in cluster_dev], dtype=int)
    cluster_eval_new = np.asarray([old_to_new[int(value)] for value in cluster_eval], dtype=int)
    cluster_candidates_new = np.asarray([old_to_new[int(value)] for value in cluster_candidates], dtype=int)
    heat = cluster_medians.loc[old_order].to_numpy(float)

    representatives = []
    for new_cluster, old_cluster in enumerate(old_order):
        members = np.flatnonzero(cluster_dev == old_cluster)
        distances = np.sum((latent_dev[members] - clusterer.cluster_centers_[old_cluster]) ** 2, axis=1)
        representative_index = int(members[int(np.argmin(distances))])
        representatives.append(
            {
                "cluster_id": f"C{new_cluster + 1:02d}",
                "domain_index": int(dev.loc[representative_index, "domain_index"]),
                "canonical_smiles": str(dev.loc[representative_index, "canonical_smiles"]),
            }
        )
    representative_table = pd.DataFrame(representatives)
    representative_csv = SOURCE / "feature_atlas_representative_structures.csv"
    representative_table.to_csv(representative_csv, index=False)
    structure_dir = CACHE / "representative_structures"
    subprocess.run(
        [str(RDKIT_PYTHON), str(Path(__file__).resolve()), "--rdkit-render", str(representative_csv), str(structure_dir)],
        check=True,
    )

    groups = {
        "Development": cluster_dev_new,
        "Evaluation": cluster_eval_new,
        "Candidates": cluster_candidates_new,
    }
    occupancy_records = []
    for group, assignments in groups.items():
        counts = np.bincount(assignments, minlength=N_CLUSTERS)
        for cluster_index, count in enumerate(counts):
            occupancy_records.append(
                {
                    "cluster_id": f"C{cluster_index + 1:02d}",
                    "group": group,
                    "count": int(count),
                    "cohort_fraction": float(count / len(assignments)),
                }
            )
    occupancy = pd.DataFrame(occupancy_records)
    occupancy.to_csv(SOURCE / "feature_atlas_cohort_occupancy.csv", index=False)
    heat_source = pd.DataFrame(heat, columns=[short for _, _, short, _ in FEATURES])
    heat_source.insert(0, "cluster_id", [f"C{i + 1:02d}" for i in range(N_CLUSTERS)])
    heat_source.to_csv(SOURCE / "feature_atlas_cluster_feature_medians.csv", index=False)
    assignments = pd.concat(
        [
            dev[["domain_index", "canonical_smiles"]].assign(group="Development", cluster=cluster_dev_new + 1),
            evaluation[["domain_index", "canonical_smiles"]].assign(group="Evaluation", cluster=cluster_eval_new + 1),
            candidates[["candidate_id", "canonical_smiles"]].rename(columns={"candidate_id": "domain_index"}).assign(
                group="Candidates", cluster=cluster_candidates_new + 1
            ),
        ],
        ignore_index=True,
    )
    assignments.to_csv(SOURCE / "feature_atlas_assignments.csv", index=False)

    cmap = LinearSegmentedColormap.from_list(
        "atlas_diverging",
        [COLORS["deep_blue"], COLORS["sky_blue"], COLORS["near_white"], COLORS["peach"], "#c87670"],
    )
    norm = TwoSlopeNorm(vmin=-2.2, vcenter=0.0, vmax=2.2)
    fig = plt.figure(figsize=(14.4, 7.7))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.75, 10.5, 2.1, 0.24], wspace=0.08)
    ax_struct = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])
    ax_occ = fig.add_subplot(grid[0, 2], sharey=ax_heat)
    ax_cbar = fig.add_subplot(grid[0, 3])

    image = ax_heat.imshow(heat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax_heat.set_xticks(np.arange(len(FEATURES)))
    ax_heat.set_xticklabels([short for _, _, short, _ in FEATURES], rotation=58, ha="right", rotation_mode="anchor")
    dev_counts = np.bincount(cluster_dev_new, minlength=N_CLUSTERS)
    ax_heat.set_yticks(np.arange(N_CLUSTERS))
    ax_heat.set_yticklabels([f"C{i + 1:02d}   n={dev_counts[i]:,}" for i in range(N_CLUSTERS)])
    ax_heat.tick_params(axis="both", length=0)
    ax_heat.set_xticks(np.arange(-0.5, len(FEATURES), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, N_CLUSTERS, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.45, alpha=0.82)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    ax_heat.set_title("Cluster-level median feature profile", loc="left", pad=25, color=COLORS["ink"], fontweight="bold")

    family_names = [family for _, _, _, family in FEATURES]
    for column, family in enumerate(family_names):
        ax_heat.add_patch(
            mpl.patches.Rectangle(
                (column - 0.5, -1.12),
                1.0,
                0.20,
                transform=ax_heat.transData,
                facecolor=FAMILY_COLORS[family],
                edgecolor="none",
                clip_on=False,
            )
        )
    family_handles = [mpl.patches.Patch(facecolor=color, label=name) for name, color in FAMILY_COLORS.items()]
    ax_heat.legend(
        handles=family_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=5,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.25,
        fontsize=7.0,
    )

    ax_struct.set_xlim(0, 1)
    ax_struct.set_ylim(N_CLUSTERS - 0.5, -0.5)
    ax_struct.axis("off")
    ax_struct.set_title("Representative structures", loc="left", pad=25, color=COLORS["ink"], fontweight="bold")
    for row_index in range(N_CLUSTERS):
        structure_path = structure_dir / f"C{row_index + 1:02d}.png"
        structure_image = plt.imread(structure_path)
        box = AnnotationBbox(
            OffsetImage(structure_image, zoom=0.25),
            (0.52, row_index),
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
        ax_struct.add_artist(box)

    group_order = ["Development", "Evaluation", "Candidates"]
    group_colors = {
        "Development": COLORS["gray_green"],
        "Evaluation": COLORS["deep_blue"],
        "Candidates": COLORS["peach"],
    }
    for x_position, group in enumerate(group_order):
        subset = occupancy.loc[occupancy["group"].eq(group)].sort_values("cluster_id")
        fractions = subset["cohort_fraction"].to_numpy(float)
        ax_occ.scatter(
            np.full(N_CLUSTERS, x_position),
            np.arange(N_CLUSTERS),
            s=18 + 720 * fractions,
            c=group_colors[group],
            edgecolors=COLORS["ink"],
            linewidths=0.45,
            alpha=0.96,
            zorder=3,
        )
    ax_occ.set_xlim(-0.55, 2.55)
    ax_occ.set_xticks(range(3))
    ax_occ.set_xticklabels(["Development\n60,641", "Evaluation\n4,000", "Candidates\n78"], rotation=32, ha="right")
    ax_occ.tick_params(axis="y", left=False, labelleft=False)
    ax_occ.tick_params(axis="x", length=0)
    ax_occ.set_title("Cohort occupancy", pad=25, color=COLORS["ink"], fontweight="bold")
    ax_occ.set_facecolor("#fbfcfc")
    for y_value in np.arange(-0.5, N_CLUSTERS, 1):
        ax_occ.axhline(y_value, color="white", linewidth=0.7, zorder=0)
    for spine in ax_occ.spines.values():
        spine.set_visible(False)
    legend_handles = [
        ax_occ.scatter([], [], s=18 + 720 * fraction, color="#cad4da", edgecolor=COLORS["ink"], linewidth=0.4, label=f"{int(fraction * 100)}%")
        for fraction in (0.05, 0.10, 0.20)
    ]
    ax_occ.legend(
        handles=legend_handles,
        title="Share of cohort",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        handletextpad=0.25,
        columnspacing=0.65,
        frameon=False,
        fontsize=6.6,
        title_fontsize=6.8,
    )

    colorbar = fig.colorbar(image, cax=ax_cbar, orientation="vertical")
    colorbar.set_ticks([-2, -1, 0, 1, 2])
    colorbar.set_label("Robust standardized\ncluster median", labelpad=7)
    colorbar.outline.set_linewidth(0.6)

    fig.suptitle(
        "Multiscale feature atlas of the molecular dataset",
        x=0.50,
        y=0.985,
        fontsize=14.2,
        fontweight="bold",
        color=COLORS["deep_blue"],
    )
    fig.text(
        0.50,
        0.014,
        "Chemical clusters, feature scaling, and row order were fitted on the development set only; evaluation and candidate molecules were projected without labels.",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=COLORS["mid_gray"],
    )
    fig.subplots_adjust(left=0.025, right=0.965, top=0.86, bottom=0.21)

    stem = OUT / "Fig_multiscale_feature_atlas"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    manifest = {
        "title": "Multiscale feature atlas of the molecular dataset",
        "development_n": len(dev),
        "development_unique_structures": int(dev["canonical_smiles"].nunique()),
        "development_duplicate_rows": int(len(dev) - dev["canonical_smiles"].nunique()),
        "evaluation_n": len(evaluation),
        "candidate_n": len(candidates),
        "n_clusters": N_CLUSTERS,
        "n_features": len(FEATURES),
        "clustering_fit_group": "development only",
        "scaling_fit_group": "development only",
        "evaluation_role": "fingerprint projection and cluster occupancy only; no labels read",
        "candidate_role": "fingerprint projection and cluster occupancy only; no predictions or labels read",
        "fingerprint": "Morgan radius 2, 1024 bits",
        "projection": "TruncatedSVD, 32 components",
        "clustering": "MiniBatchKMeans, 16 clusters",
        "heatmap_value": "cluster median after development-fitted median/IQR robust standardization; clipped to [-4, 4] before aggregation",
        "features": [
            {"source": source, "raw_name": raw, "display_name": short, "family": family}
            for source, raw, short, family in FEATURES
        ],
        "outputs": [str(stem.with_suffix(suffix)) for suffix in (".png", ".pdf", ".svg", ".tiff")],
    }
    (OUT / "Fig_multiscale_feature_atlas_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **{key: manifest[key] for key in ("development_n", "evaluation_n", "candidate_n", "n_clusters", "n_features")}}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdkit-fingerprints", nargs=2, metavar=("INPUT_CSV", "OUTPUT_NPZ"))
    parser.add_argument("--rdkit-render", nargs=2, metavar=("INPUT_CSV", "OUTPUT_DIR"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.rdkit_fingerprints:
        rdkit_fingerprint_stage(Path(arguments.rdkit_fingerprints[0]), Path(arguments.rdkit_fingerprints[1]))
    elif arguments.rdkit_render:
        rdkit_render_stage(Path(arguments.rdkit_render[0]), Path(arguments.rdkit_render[1]))
    else:
        main_plot()
