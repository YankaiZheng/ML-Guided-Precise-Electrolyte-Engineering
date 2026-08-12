#!/usr/bin/env python3
"""Build manuscript-ready Figures 2-6 for the Candidate100-clean D/P workflow.

The script only consumes frozen predictions, feature stores, and checkpoints. It
does not fit or retune a predictive model. Each figure is exported as PNG, PDF,
and SVG together with the source data needed to reproduce the plotted values.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix, vstack
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import TruncatedSVD
import torch


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "analysis_outputs/qme14s_training/domain65k"
RUNS = DOMAIN / "model_runs"
LOCKED = RUNS / "domain65k_candidate100_final_locked"
OUT = ROOT / "analysis_outputs/paper_figures_ml_workflow"
SOURCE = OUT / "source_data"

DEV_CSV = DOMAIN / "domain65k_cv_pool_complete_features.csv"
LOCKED_META_CSV = DOMAIN / "domain65k_locked_test_complete_features.csv"
CURRENT4000 = LOCKED / "candidate100_clean_locked_dft_suitability_4194_random_remove194_seed20260715_kept4000_predictions.csv"
MULTIOBJECTIVE4000 = RUNS / "domain65k_d_candidate100_fold0_fusion_multiobjective/evaluation4000_predictions.csv"
CANDIDATES = ROOT / "analysis_outputs/final_candidates_20260718/final78_source_order_predictions.csv"
TABULAR_FOLD0 = RUNS / "domain65k_d_v2_tabular_candidate100_fold0_frozen081/fold0_predictions.csv"
XGB_FOLD0 = RUNS / "domain65k_d_candidate100_xgb_fold0/fold0_predictions.csv"
VECTOR_FOLD0 = RUNS / "domain65k_d_vector_candidate100_gpu_fold0_lr5e5_e35_outer_eval_predictions.csv"
FUSION_WEIGHTS = RUNS / "domain65k_d_candidate100_fold0_fusion_multiobjective/multiobjective_weights.json"
VECTOR_CHECKPOINT = RUNS / "domain65k_d_candidate100_vector_full/full_vector.pt"
RAW_DIR = DOMAIN / "features_v2_atom3d/chunks"
OPT_H5 = ROOT / "OPT_186102.h5"
EXCLUSION = ROOT / "analysis_outputs/candidate_curation_v3/domain65k_candidate100_exact_exclusion.csv"

PYTHON = sys.executable
RNG_SEED = 20260715
KBT_EH = 3.166811563e-6 * 298.15

# Palette shared with the article's established main figure.
COLORS = {
    "deep_blue": "#4f779e",
    "sky_blue": "#91b4d1",
    "green": "#b6ccb9",
    "gray_green": "#b4c2ba",
    "mauve": "#c7b4be",
    "peach": "#e1aca6",
    "ink": "#263238",
    "mid_gray": "#7b8790",
    "light_gray": "#dce3e7",
    "near_white": "#f4f7f9",
}

MEMBERS = [
    "pred__cat_rank",
    "pred__cat_raw",
    "pred__lgbm_normal",
    "pred__lgbm_rank",
    "pred__lgbm_raw",
    "pred__xgb_rank",
    "pred__xgb_raw",
    "pred__vector_mu",
]
MEMBER_LABELS = {
    "pred__cat_rank": "CatBoost rank",
    "pred__cat_raw": "CatBoost raw",
    "pred__lgbm_normal": "LightGBM normal",
    "pred__lgbm_rank": "LightGBM rank",
    "pred__lgbm_raw": "LightGBM raw",
    "pred__xgb_rank": "XGBoost rank",
    "pred__xgb_raw": "XGBoost raw",
    "pred__vector_mu": "Equivariant vector",
    "score__fusion": "Eight-member fusion",
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
            "legend.fontsize": 6.5,
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
    plt.close(fig)


def write_manifest(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rank_pct(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / len(values)


def ndcg_at_fraction(y: np.ndarray, score: np.ndarray, fraction: float = 0.10) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    k = max(1, int(np.ceil(len(y) * fraction)))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(y)[::-1][:k]
    selected = y[np.argsort(score)[::-1][:k]]
    return float(np.dot(selected, discount) / max(np.dot(ideal, discount), 1e-12))


def top_overlap(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    k = max(1, int(np.ceil(len(y) * fraction)))
    true = set(np.argsort(y)[-k:])
    predicted = set(np.argsort(score)[-k:])
    return float(len(true & predicted) / k)


def raw_store_get(domain_index: int, cache: dict[int, dict[int, dict]]) -> dict:
    start = (int(domain_index) // 100) * 100
    if start not in cache:
        path = RAW_DIR / f"chunk_{start:06d}_{min(start + 100, 65126):06d}.pkl"
        cache[start] = {int(row["domain_index"]): row for row in pd.read_pickle(path).to_dict("records")}
    return cache[start][int(domain_index)]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Figure 2: chemical-space coverage
# ---------------------------------------------------------------------------


def fingerprints(smiles: Iterable[str], size: int = 1024) -> tuple[list, csr_matrix, list[int]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=size)
    fps = []
    rows, cols = [], []
    invalid = []
    for idx, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            invalid.append(idx)
            fp = generator.GetFingerprint(Chem.MolFromSmiles("C"))
        else:
            fp = generator.GetFingerprint(mol)
        on = list(fp.GetOnBits())
        rows.extend([idx] * len(on))
        cols.extend(on)
        fps.append(fp)
    values = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix((values, (rows, cols)), shape=(len(fps), size), dtype=np.float32)
    return fps, matrix, invalid


def nearest_tanimoto(query_fps: list, reference_fps: list) -> np.ndarray:
    result = np.empty(len(query_fps), dtype=np.float32)
    for idx, fp in enumerate(query_fps):
        result[idx] = max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps))
    return result


def figure2() -> None:
    dev = pd.read_csv(DEV_CSV, usecols=["domain_index", "canonical_smiles"], low_memory=False)
    excluded = set(pd.read_csv(EXCLUSION, usecols=["domain_index"])["domain_index"].astype(int))
    dev = dev.loc[~dev["domain_index"].astype(int).isin(excluded)].drop_duplicates("domain_index").reset_index(drop=True)
    evaluation = pd.read_csv(CURRENT4000, usecols=["domain_index", "canonical_smiles"]).reset_index(drop=True)
    candidates = pd.read_csv(CANDIDATES).reset_index(drop=True)

    cache_path = SOURCE / "fig2_chemical_space_projection.csv"
    similarity_path = SOURCE / "fig2_nearest_tanimoto.csv"
    SOURCE.mkdir(parents=True, exist_ok=True)
    cache_valid = False
    if cache_path.exists() and similarity_path.exists():
        projection = pd.read_csv(cache_path)
        similarity = pd.read_csv(similarity_path)
        cache_valid = (
            int(projection["group"].eq("Evaluation").sum()) == len(evaluation)
            and int(projection["group"].eq("Candidates").sum()) == len(candidates)
            and int(similarity["group"].eq("Evaluation").sum()) == len(evaluation)
            and int(similarity["group"].eq("Candidates").sum()) == len(candidates)
        )
    if not cache_valid:
        dev_fps, dev_matrix, invalid_dev = fingerprints(dev["canonical_smiles"])
        eval_fps, eval_matrix, invalid_eval = fingerprints(evaluation["canonical_smiles"])
        cand_fps, cand_matrix, invalid_cand = fingerprints(candidates["canonical_smiles"])
        if invalid_dev or invalid_eval or invalid_cand:
            raise RuntimeError(f"Invalid SMILES in Figure 2 inputs: dev={invalid_dev}, eval={invalid_eval}, candidates={invalid_cand}")
        combined = vstack([dev_matrix, eval_matrix, cand_matrix], format="csr")
        reducer = TruncatedSVD(n_components=2, random_state=RNG_SEED, n_iter=10)
        embedding = reducer.fit_transform(combined)
        groups = np.repeat(["Development", "Evaluation", "Candidates"], [len(dev), len(evaluation), len(candidates)])
        projection = pd.DataFrame({"latent_1": embedding[:, 0], "latent_2": embedding[:, 1], "group": groups})
        projection["domain_index"] = pd.concat(
            [dev["domain_index"], evaluation["domain_index"], pd.Series(np.nan, index=candidates.index)], ignore_index=True
        )
        projection["candidate_id"] = pd.concat(
            [pd.Series(np.nan, index=dev.index), pd.Series(np.nan, index=evaluation.index), candidates["candidate_id"]], ignore_index=True
        )
        projection.to_csv(cache_path, index=False)
        eval_similarity = nearest_tanimoto(eval_fps, dev_fps)
        cand_similarity = nearest_tanimoto(cand_fps, dev_fps)
        similarity = pd.concat(
            [
                pd.DataFrame({"group": "Evaluation", "nearest_tanimoto": eval_similarity, "id": evaluation["domain_index"]}),
                pd.DataFrame({"group": "Candidates", "nearest_tanimoto": cand_similarity, "id": candidates["candidate_id"]}),
            ],
            ignore_index=True,
        )
        similarity.to_csv(similarity_path, index=False)

    dev_xy = projection.loc[projection["group"].eq("Development")]
    eval_xy = projection.loc[projection["group"].eq("Evaluation")]
    cand_xy = projection.loc[projection["group"].eq("Candidates")]

    # Give the atlas a deliberately landscape footprint while retaining the
    # compact right-side diagnostic column.
    fig = plt.figure(figsize=(8.6, 4.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.92, 0.86], wspace=0.42)
    ax = fig.add_subplot(grid[0, 0])
    side = grid[0, 1].subgridspec(2, 1, hspace=0.42)
    ax_hist = fig.add_subplot(side[0, 0])
    ax_cov = fig.add_subplot(side[1, 0])

    atlas_cmap = LinearSegmentedColormap.from_list(
        "atlas", ["#f7fafb", "#d8e3ea", COLORS["sky_blue"], COLORS["deep_blue"]]
    )
    hb = ax.hexbin(
        dev_xy["latent_1"], dev_xy["latent_2"], gridsize=72, mincnt=1,
        cmap=atlas_cmap, bins="log", linewidths=0, rasterized=True,
    )
    x_edges = np.linspace(dev_xy["latent_1"].quantile(0.002), dev_xy["latent_1"].quantile(0.998), 100)
    y_edges = np.linspace(dev_xy["latent_2"].quantile(0.002), dev_xy["latent_2"].quantile(0.998), 100)
    hist, _, _ = np.histogram2d(eval_xy["latent_1"], eval_xy["latent_2"], bins=[x_edges, y_edges])
    smooth = gaussian_filter(hist.T, sigma=2.0)
    positive = smooth[smooth > 0]
    if len(positive):
        levels = np.unique(np.quantile(positive, [0.55, 0.78, 0.92]))
        if len(levels) >= 2:
            ax.contour(
                (x_edges[:-1] + x_edges[1:]) / 2,
                (y_edges[:-1] + y_edges[1:]) / 2,
                smooth,
                levels=levels,
                colors=[COLORS["gray_green"]] * len(levels),
                linewidths=[0.75, 1.0, 1.3][-len(levels):],
                alpha=0.95,
            )
    ax.scatter(
        cand_xy["latent_1"], cand_xy["latent_2"], s=17, c=COLORS["peach"],
        edgecolor="white", linewidth=0.45, alpha=0.95, zorder=4, label=f"{len(candidates)} candidates",
    )
    ax.set_xlabel("Morgan fingerprint latent axis 1")
    ax.set_ylabel("Morgan fingerprint latent axis 2")
    ax.set_title("Chemical-space atlas")
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(hb, ax=ax, fraction=0.038, pad=0.018)
    colorbar.set_label("Development density (log)", labelpad=2)
    panel_label(ax, "a")

    bins = np.linspace(0.15, 1.0, 35)
    for group, color in [("Evaluation", COLORS["deep_blue"]), ("Candidates", COLORS["peach"])]:
        values = similarity.loc[similarity["group"].eq(group), "nearest_tanimoto"].to_numpy(float)
        ax_hist.hist(values, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.42, linewidth=0)
        display_group = "Evaluation rows" if group == "Evaluation" else group
        ax_hist.hist(values, bins=bins, density=True, histtype="step", color=color, linewidth=1.3, label=f"{display_group} (n={len(values):,})")
        ax_hist.axvline(np.median(values), color=color, linewidth=1.0, linestyle="--")
    ax_hist.set(xlabel="Nearest-neighbor Tanimoto similarity", ylabel="Density", title="Local chemical support")
    ax_hist.xaxis.labelpad = 2
    ax_hist.yaxis.labelpad = 1
    ax_hist.legend(loc="upper left")
    panel_label(ax_hist, "b", x=-0.23)

    thresholds = np.linspace(0.2, 0.95, 151)
    coverage_rows = []
    for group, color in [("Evaluation", COLORS["deep_blue"]), ("Candidates", COLORS["peach"])]:
        values = similarity.loc[similarity["group"].eq(group), "nearest_tanimoto"].to_numpy(float)
        coverage = np.asarray([(values >= threshold).mean() for threshold in thresholds])
        coverage_rows.extend({"group": group, "threshold": t, "coverage": c} for t, c in zip(thresholds, coverage))
        ax_cov.plot(thresholds, coverage, color=color, linewidth=1.8, label=group)
    pd.DataFrame(coverage_rows).to_csv(SOURCE / "fig2_similarity_coverage_curve.csv", index=False)
    ax_cov.axvline(0.60, color=COLORS["mid_gray"], linestyle="--", linewidth=0.8)
    ax_cov.set(xlim=(0.2, 0.95), ylim=(0, 1.02), xlabel="Similarity threshold", ylabel="Coverage fraction", title="Applicability-domain coverage")
    ax_cov.xaxis.labelpad = 2
    ax_cov.yaxis.labelpad = 1
    ax_cov.legend(loc="lower left")
    panel_label(ax_cov, "c", x=-0.23)

    save_figure(fig, "Fig2_chemical_space_coverage")
    write_manifest(
        "Fig2_chemical_space_coverage",
        {
            "claim": "The evaluation and candidate molecules are mapped against the Candidate100-clean development chemical space, with local support quantified by nearest-neighbor Tanimoto similarity.",
            "n_development_unique": int((projection["group"] == "Development").sum()),
            "n_evaluation_rows": int((projection["group"] == "Evaluation").sum()),
            "n_evaluation_unique": int(evaluation["domain_index"].nunique()),
            "n_candidates": int((projection["group"] == "Candidates").sum()),
            "embedding": "two-component TruncatedSVD of radius-2, 1024-bit Morgan fingerprints",
            "nearest_neighbor_reference": "all unique Candidate100-clean development fingerprints",
            "python": PYTHON,
        },
    )


# ---------------------------------------------------------------------------
# Figure 3: vector residual correction and atom-level examples
# ---------------------------------------------------------------------------


def vector_decomposition(model, batch: dict[str, torch.Tensor], vector_module) -> dict[str, np.ndarray]:
    with torch.no_grad():
        pos, edge = batch["pos"], batch["edge_index"]
        source, target = edge[0], edge[1]
        displacement = pos[source] - pos[target]
        distance = torch.linalg.vector_norm(displacement, dim=1).clamp_min(1e-6)
        unit = displacement / distance.unsqueeze(1)
        rbf = torch.exp(-model.width * (distance.unsqueeze(1) - model.centers.to(pos)) ** 2)
        scalar = model.embedding(batch["z"].clamp_max(63)) + model.atom_input(
            torch.stack([batch["charge"], batch["formal"], batch["aromatic"]], dim=1)
        )
        equivariant = scalar.new_zeros((scalar.size(0), scalar.size(1), 3))
        for layer in model.layers:
            scalar, equivariant = layer(scalar, equivariant, edge, unit, rbf)
        n_conf = int(batch["conf_mol"].numel())
        atom_conf = batch["atom_conf"]
        q_delta = model.delta_charge(scalar).squeeze(1)
        q_delta = q_delta - vector_module.scatter_mean(q_delta, atom_conf, n_conf)[atom_conf]
        local = torch.sum(model.local_coeff(scalar).unsqueeze(-1) * equivariant, dim=1)
        charge_term = q_delta.unsqueeze(1) * pos
        conf_charge_mu = vector_module.scatter_sum(charge_term, atom_conf, n_conf)
        conf_local_mu = vector_module.scatter_sum(local, atom_conf, n_conf)
        delta_mu = conf_charge_mu + conf_local_mu
        conf_mu = batch["baseline_mu"] + delta_mu
        conf_scalar = vector_module.scatter_mean(scalar, atom_conf, n_conf)
        logits = model.attention(conf_scalar).squeeze(1) - batch["delta_energy"]
        weights = torch.softmax(logits, dim=0)
        molecular_mu = torch.sum(conf_mu * weights.unsqueeze(1), dim=0)
        baseline_mu = torch.sum(batch["baseline_mu"] * weights.unsqueeze(1), dim=0)
        charge_mu = torch.sum(conf_charge_mu * weights.unsqueeze(1), dim=0)
        local_mu = torch.sum(conf_local_mu * weights.unsqueeze(1), dim=0)
        return {
            "pos": pos.cpu().numpy(),
            "z": batch["z"].cpu().numpy(),
            "weights": weights.cpu().numpy(),
            "q_delta": q_delta.cpu().numpy(),
            "charge_term": charge_term.cpu().numpy(),
            "local_term": local.cpu().numpy(),
            "baseline_mu": baseline_mu.cpu().numpy(),
            "charge_mu": charge_mu.cpu().numpy(),
            "local_mu": local_mu.cpu().numpy(),
            "pred_mu": molecular_mu.cpu().numpy(),
        }


def aligned_target_vector(meta: pd.Series, raw: dict, align_module, h5: h5py.File) -> np.ndarray:
    group = h5[str(meta["opt_h5_key"])]
    ref_pos = np.asarray(group["pos"][()], dtype=np.float64)
    ref_z = np.asarray(group["z"][()], dtype=np.int64).reshape(-1)
    ref_mu = np.asarray(group["dipole"][()], dtype=np.float64).reshape(3)
    graph_z = np.asarray(raw["graph"]["z"], dtype=np.int64)
    energies = np.asarray([float(conf["energy_eh"]) for conf in raw["conformers"]])
    anchor = int(np.argmin(energies))
    anchor_pos = np.asarray(raw["conformers"][anchor]["pos"], dtype=np.float64)
    maps = align_module.candidate_maps(str(meta["source_smiles_y"]), str(meta["canonical_smiles"]), ref_z, graph_z)
    if not maps:
        raise RuntimeError(f"No DFT-to-xTB atom map for domain_index={meta['domain_index']}")
    best = None
    for mapping in maps:
        rotation, rmsd, _ = align_module.kabsch(ref_pos[mapping], anchor_pos, graph_z != 1)
        if best is None or rmsd < best[0]:
            best = (rmsd, rotation)
    assert best is not None
    return ref_mu @ best[1]


def aggregate_atomic_terms(decomp: dict[str, np.ndarray], n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = decomp["weights"]
    position = np.zeros((n_atoms, 3), dtype=float)
    charge = np.zeros((n_atoms, 3), dtype=float)
    local = np.zeros((n_atoms, 3), dtype=float)
    for conf, weight in enumerate(weights):
        sl = slice(conf * n_atoms, (conf + 1) * n_atoms)
        position += weight * decomp["pos"][sl]
        charge += weight * decomp["charge_term"][sl]
        local += weight * decomp["local_term"][sl]
    return position, charge, local


def project_molecule(position: np.ndarray, z: np.ndarray, vectors: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    heavy = z != 1
    centered = position - position[heavy].mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered[heavy], full_matrices=False)
    basis = vt[:2].T
    xy = centered @ basis
    projected = [np.asarray(vector) @ basis for vector in vectors]
    return xy, projected


def atom_color(atomic_number: int) -> str:
    return {
        6: "#5b6570",
        7: "#4f779e",
        8: "#d27f79",
        9: "#76a97b",
        15: "#b38a55",
        16: "#c3a24f",
        17: "#7ba78d",
        35: "#8d6f65",
    }.get(int(atomic_number), "#9aa3a9")


def project_vector_chain(record: dict) -> dict[str, np.ndarray]:
    """Project the residual vector polygon to its dominant 2D plane."""
    baseline = np.asarray(record["baseline_vec"], dtype=float)
    charge = np.asarray(record["charge_vec"], dtype=float)
    local = np.asarray(record["local_vec"], dtype=float)
    predicted = baseline + charge + local
    target = np.asarray(record["target_vec"], dtype=float)
    vectors = np.vstack([baseline, charge, local, predicted, target, baseline + charge])
    _, _, vt = np.linalg.svd(vectors, full_matrices=False)
    projected = vectors @ vt[:2].T
    target_2d = projected[4]
    angle = -float(np.arctan2(target_2d[1], target_2d[0]))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    projected = projected @ rotation.T
    if projected[4, 0] < 0:
        projected[:, 0] *= -1
    return {
        "baseline": projected[0],
        "charge": projected[1],
        "local": projected[2],
        "predicted": projected[3],
        "target": projected[4],
        "after_charge": projected[5],
    }


def draw_vector_example(ax: plt.Axes, record: dict) -> None:
    chain = project_vector_chain(record)
    origin = np.zeros(2)
    after_charge = chain["baseline"] + chain["charge"]
    predicted = after_charge + chain["local"]

    ax.axhline(0, color=COLORS["light_gray"], linewidth=0.55, zorder=0)
    ax.axvline(0, color=COLORS["light_gray"], linewidth=0.55, zorder=0)
    for start, end, color, linestyle, linewidth, zorder in [
        (origin, chain["target"], COLORS["green"], "-", 4.0, 1),
        (origin, predicted, COLORS["deep_blue"], "-", 2.1, 2),
        (origin, chain["baseline"], COLORS["mid_gray"], "--", 1.8, 4),
        (chain["baseline"], after_charge, COLORS["peach"], "-", 2.2, 5),
        (after_charge, predicted, COLORS["mauve"], "-", 2.2, 5),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10.5,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                zorder=zorder,
            )
        )
    ax.plot(
        [predicted[0], chain["target"][0]],
        [predicted[1], chain["target"][1]],
        color=COLORS["light_gray"],
        linewidth=0.8,
        linestyle=":",
        zorder=0,
    )

    ax.scatter(
        [chain["baseline"][0], after_charge[0]],
        [chain["baseline"][1], after_charge[1]],
        s=8,
        color=[COLORS["mid_gray"], COLORS["peach"]],
        zorder=6,
    )
    ax.annotate("DFT", xy=chain["target"], xytext=(0, 7), textcoords="offset points", color="#86a98c", fontsize=5.8, fontweight="bold", ha="right")
    ax.annotate("Predicted", xy=predicted, xytext=(0, -12), textcoords="offset points", color=COLORS["deep_blue"], fontsize=5.8, fontweight="bold", ha="right")
    ax.scatter([0], [0], s=11, color=COLORS["ink"], zorder=6)

    points = np.vstack([origin, chain["baseline"], after_charge, predicted, chain["target"]])
    span = max(float(np.max(np.abs(points))), 1.0)
    limit = 1.18 * span
    ax.set(xlim=(-0.28 * limit, limit), ylim=(-0.72 * limit, 0.72 * limit))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(record["label"], pad=2.5, fontweight="bold")
    ax.text(
        0.5,
        -0.03,
        f"xTB {record['baseline_mag']:.2f}  |  vector {record['pred_mag']:.2f}  |  DFT {record['target_mag']:.2f} D",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.3,
    )


def figure3() -> None:
    frame = pd.read_csv(CURRENT4000)
    raw_cache: dict[int, dict[int, dict]] = {}
    baseline_rows = []
    for row in frame.itertuples():
        raw = raw_store_get(int(row.domain_index), raw_cache)
        conformers = raw.get("conformers") or []
        if raw.get("status") != "ok" or len(conformers) != 3:
            continue
        energies = np.asarray([float(conf["energy_eh"]) for conf in conformers])
        anchor = int(np.argmin(energies))
        baseline_mag = float(conformers[anchor]["full_dipole_debye"])
        baseline_rows.append(
            {
                "domain_index": int(row.domain_index),
                "canonical_smiles": row.canonical_smiles,
                "D": float(row.D),
                "xtb_lowest_energy_mu": baseline_mag,
                "vector_mu": float(row.pred__vector_mu),
                "xtb_abs_error": abs(baseline_mag - float(row.D)),
                "vector_abs_error": abs(float(row.pred__vector_mu) - float(row.D)),
            }
        )
    errors = pd.DataFrame(baseline_rows)
    errors["absolute_error_gain"] = errors["xtb_abs_error"] - errors["vector_abs_error"]
    errors.to_csv(SOURCE / "fig3_vector_correction_all4000.csv", index=False)

    vector_module = load_module("paper_vector_model", ROOT / "analysis_outputs/run_domain65k_d_vector_equivariant.py")
    align_module = load_module("paper_vector_alignment", ROOT / "analysis_outputs/build_domain65k_d_vector_alignment.py")
    checkpoint = torch.load(VECTOR_CHECKPOINT, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    model = vector_module.XTBVectorDipole(hidden=int(args.get("hidden", 128)), layers=int(args.get("layers", 4)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    locked_meta = pd.read_csv(LOCKED_META_CSV, low_memory=False).drop_duplicates("domain_index").set_index("domain_index")
    selected = [
        (37232, "Cyclic carbonate"),
        (37290, "Cyclic sulfone"),
        (48041, "Fluorinated nitrile"),
    ]
    example_records = []
    atom_rows = []
    with h5py.File(OPT_H5, "r") as h5:
        for domain_index, label in selected:
            meta = locked_meta.loc[domain_index]
            raw = raw_store_get(domain_index, raw_cache)
            energies = np.asarray([float(conf["energy_eh"]) for conf in raw["conformers"]])
            item = {
                "domain_index": domain_index,
                "D": float(meta["D"]),
                "target_mu": np.zeros(3, dtype=np.float32),
                "rank_target": 0.0,
                "anchor": int(np.argmin(energies)),
                "raw": raw,
            }
            batch = vector_module.collate([item])
            decomp = vector_decomposition(model, batch, vector_module)
            target = aligned_target_vector(meta, raw, align_module, h5)
            n_atoms = len(raw["graph"]["z"])
            position, charge, local = aggregate_atomic_terms(decomp, n_atoms)
            total = charge + local
            xy, projected = project_molecule(position, np.asarray(raw["graph"]["z"]), [decomp["baseline_mu"], decomp["pred_mu"], target] + [v for v in total])
            baseline_2d, pred_2d, target_2d = projected[:3]
            atomic_total_2d = np.asarray(projected[3:])
            record = {
                "domain_index": domain_index,
                "label": label,
                "smiles": str(meta["canonical_smiles"]),
                "xy": xy,
                "z": np.asarray(raw["graph"]["z"]),
                "edge_index": np.asarray(raw["graph"]["edge_index"]),
                "atomic_total_2d": atomic_total_2d,
                "baseline_2d": baseline_2d,
                "pred_2d": pred_2d,
                "target_2d": target_2d,
                "baseline_vec": np.asarray(decomp["baseline_mu"]),
                "charge_vec": np.asarray(decomp["charge_mu"]),
                "local_vec": np.asarray(decomp["local_mu"]),
                "pred_vec": np.asarray(decomp["pred_mu"]),
                "target_vec": np.asarray(target),
                "baseline_mag": float(np.linalg.norm(decomp["baseline_mu"])),
                "pred_mag": float(np.linalg.norm(decomp["pred_mu"])),
                "target_mag": float(np.linalg.norm(target)),
            }
            example_records.append(record)
            for atom_index, (atomic_number, q_vec, local_vec, total_vec) in enumerate(zip(record["z"], charge, local, total)):
                atom_rows.append(
                    {
                        "domain_index": domain_index,
                        "label": label,
                        "canonical_smiles": str(meta["canonical_smiles"]),
                        "atom_index": atom_index,
                        "element": Chem.GetPeriodicTable().GetElementSymbol(int(atomic_number)),
                        "charge_correction_norm": float(np.linalg.norm(q_vec)),
                        "local_dipole_norm": float(np.linalg.norm(local_vec)),
                        "total_correction_norm": float(np.linalg.norm(total_vec)),
                    }
                )
    pd.DataFrame(atom_rows).to_csv(SOURCE / "fig3_representative_atomic_corrections.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.85))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.88], hspace=0.34, wspace=0.12)
    ax_ecdf = fig.add_subplot(grid[0, :2])
    ax_gain = fig.add_subplot(grid[0, 2])
    mol_axes = [fig.add_subplot(grid[1, idx]) for idx in range(3)]

    for column, label, color in [
        ("xtb_abs_error", "xTB baseline", COLORS["mid_gray"]),
        ("vector_abs_error", "Equivariant vector", COLORS["deep_blue"]),
    ]:
        values = np.sort(errors[column].to_numpy(float))
        cumulative = np.arange(1, len(values) + 1) / len(values)
        ax_ecdf.plot(values, cumulative, color=color, linewidth=1.9, label=f"{label}  median={np.median(values):.2f} D")
    ax_ecdf.set(xlim=(0, np.quantile(errors["xtb_abs_error"], 0.98)), ylim=(0, 1.01), xlabel="Absolute dipole-magnitude error (D)", ylabel="Cumulative fraction", title="Vector residual decomposition improves the xTB baseline")
    ax_ecdf.legend(loc="lower right")
    panel_label(ax_ecdf, "a", x=-0.08)

    gain = errors["absolute_error_gain"].to_numpy(float)
    ax_gain.hist(gain, bins=np.linspace(np.quantile(gain, 0.01), np.quantile(gain, 0.99), 38), color=COLORS["sky_blue"], edgecolor="white", linewidth=0.25)
    ax_gain.axvline(0, color=COLORS["ink"], linewidth=0.9, linestyle="--")
    improved = float((gain > 0).mean())
    ax_gain.text(0.97, 0.94, f"Improved: {improved:.1%}", transform=ax_gain.transAxes, ha="right", va="top", fontweight="bold", color=COLORS["deep_blue"])
    ax_gain.set(xlabel="Reduction in absolute error (D)", ylabel="Molecules", title="Test-wide correction gain")
    panel_label(ax_gain, "b", x=-0.21)

    for idx, (ax, record) in enumerate(zip(mol_axes, example_records)):
        draw_vector_example(ax, record)
        panel_label(ax, chr(ord("c") + idx), x=-0.02, y=1.02)
    handles = [
        Line2D([0], [0], color=COLORS["mid_gray"], linestyle="--", linewidth=1.6, label="xTB baseline vector"),
        Line2D([0], [0], color=COLORS["peach"], linewidth=1.8, label=r"Charge correction $\Delta\mu_q$"),
        Line2D([0], [0], color=COLORS["mauve"], linewidth=1.8, label=r"Local correction $\Delta\mu_{local}$"),
        Line2D([0], [0], color=COLORS["deep_blue"], linewidth=1.6, label="Predicted vector"),
        Line2D([0], [0], color=COLORS["green"], linewidth=3.0, label="DFT reference vector"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=5.7)
    save_figure(fig, "Fig3_vector_residual_correction")
    write_manifest(
        "Fig3_vector_residual_correction",
        {
            "claim": "The equivariant member learns charge-displacement and local-dipole residual vectors that add to the xTB baseline and recover the DFT dipole.",
            "n_evaluation": len(errors),
            "median_xtb_absolute_error": float(errors["xtb_abs_error"].median()),
            "median_vector_absolute_error": float(errors["vector_abs_error"].median()),
            "fraction_improved": improved,
            "representative_domain_indices": [item[0] for item in selected],
            "selection_note": "Three compact electrolyte-relevant motif classes with vector absolute error below 0.25 D and baseline-error reduction above 3 D.",
            "vector_view": "A common linear projection of each residual vector polygon preserves vector addition; the projected DFT vector is aligned horizontally for comparison.",
            "interpretation_guardrail": "Charge and local arrows are learned architecture components, not uniquely identifiable quantum-mechanical atomic observables.",
            "python": PYTHON,
        },
    )


# ---------------------------------------------------------------------------
# Figure 4: member complementarity and fusion
# ---------------------------------------------------------------------------


def fold0_member_frame() -> tuple[pd.DataFrame, dict[str, float]]:
    # Eight protocol records duplicate an identical molecular identity and carry
    # identical tabular/XGB predictions. The graph-valid vector table contains
    # one prediction per identity, so collapse those exact duplicates before
    # constructing the 12,114-row common validation surface.
    tabular = pd.read_csv(TABULAR_FOLD0).drop_duplicates("domain_index", keep="first")
    xgb = pd.read_csv(XGB_FOLD0).drop_duplicates("domain_index", keep="first")
    vector = pd.read_csv(VECTOR_FOLD0)
    frame = tabular.merge(xgb[["domain_index", "pred__xgb_raw", "pred__xgb_rank"]], on="domain_index", how="inner", validate="many_to_one")
    frame = frame.merge(vector[["domain_index", "pred__vector_mu"]], on="domain_index", how="inner", validate="many_to_one")
    payload = json.loads(FUSION_WEIGHTS.read_text(encoding="utf-8"))
    weights = {name: float(weight) for name, weight in payload["weights_by_member"].items()}
    frame["score__fusion"] = sum(weights[name] * rank_pct(frame[name].to_numpy(float)) for name in MEMBERS)
    return frame, weights


def figure4() -> None:
    frame, weights = fold0_member_frame()
    y = frame["D"].to_numpy(float)
    true_rank = rank_pct(y)
    residuals = {}
    metrics = []
    for member in MEMBERS + ["score__fusion"]:
        score = frame[member].to_numpy(float)
        pred_rank = rank_pct(score)
        residuals[member] = pred_rank - true_rank
        metrics.append(
            {
                "member": member,
                "label": MEMBER_LABELS[member],
                "spearman": float(spearmanr(y, score).statistic),
                "ndcg_at_10pct": ndcg_at_fraction(y, score, 0.10),
            }
        )
    metric_frame = pd.DataFrame(metrics)
    residual_frame = pd.DataFrame(residuals)
    correlation = residual_frame[MEMBERS].corr(method="pearson")
    metric_frame.to_csv(SOURCE / "fig4_fold0_member_metrics.csv", index=False)
    correlation.rename(index=MEMBER_LABELS, columns=MEMBER_LABELS).to_csv(SOURCE / "fig4_rank_error_correlation.csv")
    pd.DataFrame({"member": MEMBERS, "label": [MEMBER_LABELS[x] for x in MEMBERS], "weight": [weights[x] for x in MEMBERS]}).to_csv(SOURCE / "fig4_fusion_weights.csv", index=False)

    compact_labels = {
        "pred__cat_rank": "CatBoost rank",
        "pred__cat_raw": "CatBoost raw",
        "pred__lgbm_normal": "LGB normal",
        "pred__lgbm_rank": "LGB rank",
        "pred__lgbm_raw": "LGB raw",
        "pred__xgb_rank": "XGB rank",
        "pred__xgb_raw": "XGB raw",
        "pred__vector_mu": "Vector",
        "score__fusion": "Fusion",
    }
    plot_metrics = metric_frame.sort_values("spearman", ascending=False).reset_index(drop=True)
    x_pos = np.arange(len(plot_metrics))
    bar_width = 0.36
    fusion_mask = plot_metrics["member"].eq("score__fusion").to_numpy()
    spearman_colors = [COLORS["deep_blue"] if is_fusion else COLORS["sky_blue"] for is_fusion in fusion_mask]
    ndcg_colors = ["#88a98e" if is_fusion else COLORS["green"] for is_fusion in fusion_mask]
    fig, ax_metric = plt.subplots(figsize=(7.2, 3.45))
    ax_metric.axvspan(-0.48, 0.48, color=COLORS["near_white"], zorder=0)
    spearman_bars = ax_metric.bar(
        x_pos - bar_width / 2,
        plot_metrics["spearman"],
        width=bar_width,
        color=spearman_colors,
        label="Spearman",
        zorder=3,
    )
    ndcg_bars = ax_metric.bar(
        x_pos + bar_width / 2,
        plot_metrics["ndcg_at_10pct"],
        width=bar_width,
        color=ndcg_colors,
        label="NDCG@10%",
        zorder=3,
    )
    ax_metric.bar_label(spearman_bars, labels=[f"{value:.3f}" for value in plot_metrics["spearman"]], padding=2, fontsize=5.2)
    ax_metric.bar_label(ndcg_bars, labels=[f"{value:.3f}" for value in plot_metrics["ndcg_at_10pct"]], padding=2, fontsize=5.2)
    ax_metric.set_xticks(x_pos, [compact_labels[item] for item in plot_metrics["member"]], rotation=28, ha="right")
    ax_metric.set_ylim(0, 1.0)
    ax_metric.set_ylabel("Validation score")
    ax_metric.set_title(f"Development validation on fold 0 (n={len(frame):,})", pad=24)
    ax_metric.grid(axis="y", color=COLORS["light_gray"], linewidth=0.55, alpha=0.8)
    ax_metric.set_axisbelow(True)
    ax_metric.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2)
    panel_label(ax_metric, "c", x=-0.06, y=1.08)

    save_figure(fig, "Fig4_member_complementarity_fusion")
    fusion_row = metric_frame.loc[metric_frame["member"].eq("score__fusion")].iloc[0]
    write_manifest(
        "Fig4_member_complementarity_fusion",
        {
            "claim": "The frozen fusion jointly improves Spearman rank correlation and NDCG@10% over every individual member on the common fold-0 validation set.",
            "n_common_graph_valid_fold0": len(frame),
            "fusion_spearman": float(fusion_row["spearman"]),
            "fusion_ndcg_at_10pct": float(fusion_row["ndcg_at_10pct"]),
            "weights": weights,
            "locked_labels_used": False,
            "python": PYTHON,
        },
    )


# ---------------------------------------------------------------------------
# Figure 5: final D ranking density and screening utility
# ---------------------------------------------------------------------------


def figure5() -> None:
    frame = pd.read_csv(MULTIOBJECTIVE4000)
    y = frame["D"].to_numpy(float)
    score = frame["score__final_fusion_multiobjective"].to_numpy(float)
    true_rank = rankdata(-y, method="average")
    pred_rank = rankdata(-score, method="average")
    rho = float(spearmanr(y, score).statistic)

    fig = plt.figure(figsize=(7.2, 4.05))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.92], wspace=0.31)
    ax_density = fig.add_subplot(grid[0, 0])
    ax_gain = fig.add_subplot(grid[0, 1])

    density_bounds = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    density_cmap = ListedColormap(["#edf3f7", "#cad9e5", "#9fb9cd", "#7395b1", "#486d89"])
    density_norm = BoundaryNorm(density_bounds, density_cmap.N, clip=True)
    density = ax_density.hexbin(
        true_rank, pred_rank, gridsize=72, extent=(1, 4000, 1, 4000), mincnt=2,
        norm=density_norm, cmap=density_cmap, linewidths=0, edgecolors="none", rasterized=True,
    )
    diagonal = np.linspace(1, 4000, 400)
    ax_density.plot(diagonal, diagonal, color=COLORS["ink"], linestyle="--", linewidth=1.0, alpha=0.68, label="Perfect ranking")
    slope, intercept = np.polyfit(true_rank, pred_rank, 1)
    ax_density.plot(diagonal, slope * diagonal + intercept, color=COLORS["deep_blue"], linewidth=1.55, label="Linear rank fit")
    ticks = [1, 1000, 2000, 3000, 4000]
    ax_density.set(xlim=(1, 4000), ylim=(1, 4000), xticks=ticks, yticks=ticks, xlabel="True D rank (1 = highest)", ylabel="Predicted D rank (1 = highest)", title=f"Ranking consistency  |  Spearman ρ = {rho:.3f}")
    ax_density.set_aspect("equal")
    ax_density.legend(loc="upper left")
    colorbar = fig.colorbar(density, ax=ax_density, fraction=0.046, pad=0.022)
    colorbar.ax.set_title("Count", fontsize=6.2, pad=3)
    colorbar.set_ticks([2, 3, 4, 5, 6], labels=["2", "3", "4", "5", "≥6"])
    panel_label(ax_density, "a", x=-0.17)

    methods = [
        ("score__final_fusion_multiobjective", "Eight-member fusion", COLORS["deep_blue"], 2.1),
        ("score__multiobjective__vector_mu", "Equivariant vector", COLORS["green"], 1.45),
        ("score__multiobjective__xgb_rank", "XGBoost rank", COLORS["sky_blue"], 1.35),
        ("score__multiobjective__lgbm_rank", "LightGBM rank", COLORS["mauve"], 1.35),
    ]
    fractions = np.linspace(0.01, 0.50, 100)
    target_fraction = 0.10
    true_top = set(np.argsort(y)[-int(np.ceil(len(y) * target_fraction)):])
    curve_rows = []
    for column, label, color, linewidth in methods:
        member_score = frame[column].to_numpy(float)
        order = np.argsort(member_score)[::-1]
        recall = []
        for fraction in fractions:
            k = max(1, int(np.ceil(len(y) * fraction)))
            value = len(true_top & set(order[:k])) / len(true_top)
            recall.append(value)
            curve_rows.append({"method": label, "screened_fraction": fraction, "recall_of_true_top10pct": value})
        ax_gain.plot(fractions * 100, np.asarray(recall) * 100, color=color, linewidth=linewidth, label=label)
    ax_gain.plot(fractions * 100, fractions * 100, color=COLORS["mid_gray"], linestyle="--", linewidth=0.9, label="Random expectation")
    pd.DataFrame(curve_rows).to_csv(SOURCE / "fig5_top10_screening_curves.csv", index=False)
    fusion_recall10 = top_overlap(y, score, 0.10)
    ax_gain.scatter([10], [fusion_recall10 * 100], s=34, c=COLORS["deep_blue"], edgecolor="white", linewidth=0.6, zorder=5)
    ax_gain.text(10.8, fusion_recall10 * 100 + 1.5, f"{fusion_recall10:.1%} recovered", color=COLORS["deep_blue"], fontweight="bold")
    ax_gain.set(xlim=(0, 50), ylim=(0, 102), xlabel="Evaluation set screened (%)", ylabel="True top-10% recall (%)", title="Practical top-candidate recovery")
    ax_gain.legend(loc="lower right")
    panel_label(ax_gain, "b", x=-0.17)

    pd.DataFrame({"domain_index": frame["domain_index"], "D": y, "fusion_score": score, "true_rank": true_rank, "predicted_rank": pred_rank}).to_csv(SOURCE / "fig5_rank_density_source.csv", index=False)
    save_figure(fig, "Fig5_D_ranking_and_screening")
    write_manifest(
        "Fig5_D_ranking_and_screening",
        {
            "claim": "The frozen D ensemble preserves global rank order and recovers a majority of the truly highest-D molecules within a practical 10% screen.",
            "n_evaluation": len(frame),
            "spearman": rho,
            "top10_overlap": fusion_recall10,
            "fusion_protocol": "Fold-0 Pareto max-min compromise between Spearman and NDCG@10%",
            "prediction_column": "score__final_fusion_multiobjective",
            "density_scale": "five fully opaque blue-gray levels for counts 2, 3, 4, 5, and >=6; singleton bins omitted",
            "subset": "fixed random 4,000-row evaluation subset",
            "python": PYTHON,
        },
    )


# ---------------------------------------------------------------------------
# Figure 6: candidate decision landscape and member consensus
# ---------------------------------------------------------------------------


def short_candidate_name(row: pd.Series) -> str:
    text = str(row.get("standard_name", ""))
    smiles = str(row.get("canonical_smiles", ""))
    replacements = {
        "Trimethylene sulfate": "TMS",
        "5-Methyl-1,3-dioxan-2-one": "5-Me-DOC",
        "Sulfolane": "Sulfolane",
        "3-Methylsulfolane": "3-Me-sulfolane",
        "Triethyl phosphate": "TEP",
        "3-Butoxypropionitrile": "BPN",
        "Vinyl ethylene carbonate": "VEC",
        "Gamma-valerolactone": "GVL",
        "Ethyl methyl sulfone": "EtMe sulfone",
        "Diethyl sulfone": "Et2 sulfone",
        "Dimethyl sulfone": "Me2 sulfone",
    }
    if text in replacements:
        return replacements[text]
    smiles_names = {
        "O=S1(=O)OCCCO1": "TMS",
        "O=S1(=O)CCCC1": "Sulfolane",
        "O=C1OCC(C)O1": "PC",
        "O=C1OCCO1": "EC",
        "O=S1(=O)OCCCO1": "TMS",
        "O=S1(=O)OCCCO1": "TMS",
    }
    if smiles in smiles_names:
        return smiles_names[smiles]
    if "DMTMSA" in text:
        return "DMTMSA"
    if "EHFB" in text:
        return "EHFB"
    if "DMOTFS" in text:
        return "DMOTFS"
    if "TRIGLYME" in text:
        return "TRIGLYME"
    if "EGBE" in text:
        return "EGBE"
    if "丁内酯" in text or smiles == "O=C1CCCO1":
        return "GBL"
    if "FEC" in text or "氟代碳酸乙烯酯" in text:
        return "FEC"
    if "PC" in text or "碳酸丙烯酯" in text:
        return "PC"
    if "EC" in text or "碳酸乙烯酯" in text:
        return "EC"
    if "PS" in text or "丙烷磺内酯" in text:
        return "PS"
    return text if text and text != "nan" else smiles


def figure6() -> None:
    frame = pd.read_csv(CANDIDATES)
    frame["D_percentile"] = frame["D_score__final_eight_member"] * 100
    frame["short_name"] = frame.apply(short_candidate_name, axis=1)
    frame.to_csv(SOURCE / "fig6_candidate_decision_landscape.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.75))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.24, 0.92], wspace=0.34)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_consensus = fig.add_subplot(grid[0, 1])

    grade_style = {
        "original_candidate": ("o", COLORS["sky_blue"], "Original 46"),
        "direct": ("s", COLORS["green"], "Direct literature evidence"),
        "family": ("D", COLORS["mauve"], "Family-level evidence"),
    }
    for grade, (marker, color, label) in grade_style.items():
        subset = frame.loc[frame["evidence_grade"].eq(grade)]
        ax_map.scatter(
            subset["P_pred__single_lgbm_raw"], subset["D_percentile"],
            s=28, marker=marker, c=color, edgecolor="white", linewidth=0.55, alpha=0.92, label=f"{label} (n={len(subset)})",
        )
    pareto = frame.loc[frame["D_pareto_P"].astype(bool)].sort_values("P_pred__single_lgbm_raw")
    ax_map.plot(pareto["P_pred__single_lgbm_raw"], pareto["D_percentile"], color=COLORS["ink"], linewidth=1.0, linestyle="--", alpha=0.65, zorder=1)
    ax_map.scatter(pareto["P_pred__single_lgbm_raw"], pareto["D_percentile"], s=54, facecolor="none", edgecolor=COLORS["ink"], linewidth=1.0, zorder=5)

    # The consensus panel already names all top-15 candidates. Keep the map
    # readable by labeling only chemically recognizable anchors and the
    # well-separated Pareto points.
    map_labels = {
        "TMS", "PC", "Sulfolane", "3-Me-sulfolane", "EHFB", "BPN",
        "DMOTFS", "TEP", "TRIGLYME", "EGBE", "DMTMSA",
    }
    label_rows = frame.loc[frame["short_name"].isin(map_labels)].copy()
    offsets = {
        "TMS": (8, -14),
        "PC": (-8, -13),
        "Sulfolane": (-8, -17),
        "3-Me-sulfolane": (8, -18),
        "EHFB": (8, 8),
        "BPN": (8, -10),
        "DMOTFS": (-8, 11),
        "TEP": (8, 8),
        "TRIGLYME": (8, -9),
        "EGBE": (-8, -11),
    }
    for _, row in label_rows.iterrows():
        x, y = row["P_pred__single_lgbm_raw"], row["D_percentile"]
        dx, dy = offsets.get(row["short_name"], (6, 6))
        ax_map.annotate(
            row["short_name"], (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=5.8, ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top",
            arrowprops={"arrowstyle": "-", "color": "#aab3b8", "lw": 0.45},
        )
    ax_map.set(xlabel="Predicted P", ylabel="D ensemble screening percentile", title="D-P candidate decision landscape")
    ax_map.set_ylim(0, 103)
    ax_map.legend(loc="lower left")
    panel_label(ax_map, "a", x=-0.14)

    member_ranks = pd.DataFrame(index=frame.index)
    for member in MEMBERS:
        member_ranks[member] = rankdata(-frame[member].to_numpy(float), method="average")
    top = frame.nsmallest(15, "D_rank_desc").copy()
    top_indices = top.index.to_numpy()
    rank_values = member_ranks.loc[top_indices, MEMBERS].to_numpy(float)
    top["member_rank_min"] = rank_values.min(axis=1)
    top["member_rank_q25"] = np.quantile(rank_values, 0.25, axis=1)
    top["member_rank_median"] = np.median(rank_values, axis=1)
    top["member_rank_q75"] = np.quantile(rank_values, 0.75, axis=1)
    top["member_rank_max"] = rank_values.max(axis=1)
    top.to_csv(SOURCE / "fig6_top15_member_rank_consensus.csv", index=False)
    top = top.sort_values("D_rank_desc", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(top))
    for idx, row in top.iterrows():
        color = COLORS["deep_blue"] if bool(row["D_pareto_P"]) else COLORS["sky_blue"]
        ax_consensus.plot([row.member_rank_min, row.member_rank_max], [idx, idx], color="#c4cdd2", linewidth=1.0)
        ax_consensus.plot([row.member_rank_q25, row.member_rank_q75], [idx, idx], color=color, linewidth=4.0, solid_capstyle="round")
        ax_consensus.scatter([row.D_rank_desc], [idx], s=24, c=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax_consensus.set_yticks(y_pos, top["short_name"])
    ax_consensus.set_xlim(0, min(100, max(60, float(top["member_rank_max"].max()) + 3)))
    ax_consensus.set_xlabel("Rank among 100 candidates")
    ax_consensus.set_title("Eight-member rank consensus")
    ax_consensus.text(0.98, 0.02, "line: full range\nthick: interquartile range\ndot: fused rank", transform=ax_consensus.transAxes, ha="right", va="bottom", fontsize=5.8, color=COLORS["mid_gray"])
    panel_label(ax_consensus, "b", x=-0.24)

    save_figure(fig, "Fig6_candidate_Pareto_and_consensus")
    write_manifest(
        "Fig6_candidate_Pareto_and_consensus",
        {
            "claim": "The 100-candidate screen exposes a D-P trade-off and identifies a compact Pareto frontier while showing the cross-member stability of the highest-ranked D candidates.",
            "n_candidates": len(frame),
            "n_pareto": int(frame["D_pareto_P"].astype(bool).sum()),
            "member_rank_consensus": "range and interquartile range across all eight frozen D members",
            "python": PYTHON,
        },
    )


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    figure2()
    print("completed Figure 2", flush=True)
    figure3()
    print("completed Figure 3", flush=True)
    figure4()
    print("completed Figure 4", flush=True)
    figure5()
    print("completed Figure 5", flush=True)
    figure6()
    print("completed Figure 6", flush=True)
    write_manifest(
        "figure_set",
        {
            "status": "complete",
            "figures": [
                "Fig2_chemical_space_coverage",
                "Fig3_vector_residual_correction",
                "Fig4_member_complementarity_fusion",
                "Fig5_D_ranking_and_screening",
                "Fig6_candidate_Pareto_and_consensus",
            ],
            "palette": COLORS,
            "formats": ["png", "pdf", "svg"],
            "source_data_directory": str(SOURCE),
            "python": PYTHON,
        },
    )


if __name__ == "__main__":
    main()
