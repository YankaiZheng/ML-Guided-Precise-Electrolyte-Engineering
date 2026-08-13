#!/usr/bin/env python3
"""Validate public D-P Compass data tables and reported ranking metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURE_SOURCE = DATA / "figure_source"
OUT = ROOT / "results"


def ndcg_at_10pct(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """NDCG@10% with raw physical labels as gains, matching the manuscript."""
    n_top = max(1, int(np.ceil(0.10 * len(y_true))))
    order = np.argsort(-y_score, kind="mergesort")[:n_top]
    ideal = np.argsort(-y_true, kind="mergesort")[:n_top]
    discounts = 1.0 / np.log2(np.arange(2, n_top + 2))
    return float(np.sum(y_true[order] * discounts) / np.sum(y_true[ideal] * discounts))


def spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Spearman correlation without a SciPy dependency for release validation."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    true_rank = pd.Series(y_true).rank(method="average").to_numpy(float)
    score_rank = pd.Series(y_score).rank(method="average").to_numpy(float)
    return float(np.corrcoef(true_rank, score_rank)[0, 1])


def check_close(name: str, actual: float, expected: float, atol: float = 1e-10) -> float:
    if not np.isclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{name}: expected {expected:.12f}, got {actual:.12f}")
    return actual


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(DATA / "test" / "test4000_predictions.csv")
    if len(test) != 4000 or test["canonical_smiles"].nunique() != 3998:
        raise AssertionError("Test-4000 must contain 4,000 records and 3,998 unique canonical SMILES")
    d_score = test["score__final_fusion_multiobjective"].to_numpy(float)
    p_score = test["P_pred__single_lgbm_raw"].to_numpy(float)
    metrics = {
        "test4000": {
            "records": int(len(test)),
            "unique_canonical_smiles": int(test["canonical_smiles"].nunique()),
            "D": {
                "score_column": "score__final_fusion_multiobjective",
                "spearman": check_close("Test-4000 D Spearman", spearman(test["D"], d_score), 0.8357413560780362),
                "ndcg_at_10pct": check_close("Test-4000 D NDCG@10%", ndcg_at_10pct(test["D"].to_numpy(float), d_score), 0.8976287222430712),
            },
            "P": {
                "score_column": "P_pred__single_lgbm_raw",
                "spearman": check_close("Test-4000 P Spearman", spearman(test["P"], p_score), 0.9969179177907334),
                "ndcg_at_10pct": check_close("Test-4000 P NDCG@10%", ndcg_at_10pct(test["P"].to_numpy(float), p_score), 0.9981411886668111),
            },
        }
    }

    broad = pd.read_csv(DATA / "external" / "external_broad_n140.csv")
    strict = pd.read_csv(DATA / "external" / "external_strict_n34.csv")
    if len(broad) != 140 or len(strict) != 34:
        raise AssertionError("External cohorts must contain 140 broad and 34 strict records")
    if not set(strict["canonical_smiles"]).issubset(set(broad["canonical_smiles"])):
        raise AssertionError("The strict external subset must be nested within the broad cohort")
    external_metrics: dict[str, dict[str, float | int]] = {}
    for name, cohort, expected_spearman, expected_ndcg in (
        ("broad_n140", broad, 0.790796716920716, 0.8961863747532143),
        ("strict_n34", strict, 0.8563789152024445, 0.9395083234807267),
    ):
        y = cohort["D"].to_numpy(float)
        score = cohort["D_score__recomputed_subset"].to_numpy(float)
        external_metrics[name] = {
            "records": int(len(cohort)),
            "score_column": "D_score__recomputed_subset",
            "spearman": check_close(f"External {name} D Spearman", spearman(y, score), expected_spearman),
            "ndcg_at_10pct": check_close(f"External {name} D NDCG@10%", ndcg_at_10pct(y, score), expected_ndcg),
        }
    metrics["external_D"] = external_metrics

    fusion_manifest = json.loads(
        (ROOT / "code" / "frozen_configs" / "final_d_fusion_weights.json").read_text(encoding="utf-8")
    )
    weights = pd.Series(fusion_manifest["weights_by_member"], dtype=float)
    if len(weights) != 8 or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise AssertionError("Frozen D fusion must have eight weights summing to one")
    metrics["frozen_d_fusion"] = {"members": int(len(weights)), "weight_sum": float(weights.sum())}

    candidates = pd.read_csv(DATA / "candidates" / "candidate78_final_predictions.csv")
    knee = pd.read_csv(DATA / "candidates" / "candidate78_pareto_knee.csv")
    selected = knee.loc[knee["selected"].astype(bool)]
    if len(candidates) != 78 or len(knee) != 8 or len(selected) != 1:
        raise AssertionError("The candidate release must contain 78 candidates and one selected knee among eight frontier points")
    row = selected.iloc[0]
    if row["standard_name"] != "DMTMSA":
        raise AssertionError("The frozen selected candidate must be DMTMSA")
    metrics["candidate_selection"] = {
        "candidates": int(len(candidates)),
        "pareto_frontier": int(len(knee)),
        "selected": str(row["standard_name"]),
        "knee_distance": check_close("DMTMSA knee distance", float(row["knee_distance"]), 0.321044, atol=1e-6),
    }

    index = pd.read_csv(ROOT / "metadata" / "figure_source_index.csv")
    released_sources = {path.name for path in FIGURE_SOURCE.glob("*.csv")}
    missing_sources = sorted(set(index["source_file"]) - released_sources)
    if missing_sources:
        raise AssertionError(f"Figure source data missing from the release: {missing_sources}")
    metrics["figure_source"] = {
        "indexed_source_rows": int(len(index)),
        "released_csv_files": int(len(released_sources)),
        "missing_files": missing_sources,
    }

    algorithm_table = pd.read_csv(FIGURE_SOURCE / "fig2b_algorithm_evaluation4000.csv")
    if "TabM" in set(algorithm_table["model"]):
        raise AssertionError("The withdrawn TabM diagnostic must not appear in the published comparison table")
    formal_row = algorithm_table.loc[algorithm_table["model"].eq("Final fusion")]
    if len(formal_row) != 1:
        raise AssertionError("The public algorithm table must have one final-fusion row")
    metrics["algorithm_comparison"] = {
        "models": int(len(algorithm_table)),
        "withdrawn_tabm_included": False,
        "final_fusion_spearman": check_close(
            "Figure 2b final-fusion Spearman", float(formal_row.iloc[0]["spearman"]), metrics["test4000"]["D"]["spearman"]
        ),
        "final_fusion_ndcg_at_10pct": check_close(
            "Figure 2b final-fusion NDCG@10%", float(formal_row.iloc[0]["ndcg_at_10pct"]), metrics["test4000"]["D"]["ndcg_at_10pct"]
        ),
    }

    (OUT / "public_release_validation.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
