#!/usr/bin/env python3
"""Apply the fold-0-frozen multiobjective fusion to held-out predictions.

Member ranks are computed over the original 4,277-row Candidate100-clean
inference cohort, matching the frozen final-inference contract.  The fixed
4,000-row chemistry-curated evaluation cohort is selected only afterwards.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from run_domain65k_p_candidate100_single_lgbm import ranking_metrics


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs"
LOCKED_DIR = RUN / "domain65k_candidate100_final_locked"
FUSION_DIR = RUN / "domain65k_d_candidate100_fold0_fusion_multiobjective"
WEIGHTS = FUSION_DIR / "multiobjective_weights.json"
LOCKED_4277 = LOCKED_DIR / "candidate100_clean_locked_predictions.csv"
COHORT_4000 = LOCKED_DIR / "candidate100_clean_locked_dft_suitability_4194_random_remove194_seed20260715_kept4000_predictions.csv"
OUT_4277 = FUSION_DIR / "candidate100_clean_locked4277_predictions.csv"
OUT_4000 = FUSION_DIR / "evaluation4000_predictions.csv"
OUT_METRICS = FUSION_DIR / "evaluation4000_metrics.csv"


def rank01(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average").astype(np.float64) / len(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    payload = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    if payload.get("status") != "fold0_multiobjective_fusion_frozen":
        raise RuntimeError("Multiobjective fold-0 weights are not frozen")
    weights = {str(name): float(value) for name, value in payload["weights_by_member"].items()}
    if any(value < 0 for value in weights.values()) or not np.isclose(sum(weights.values()), 1.0):
        raise RuntimeError("Invalid fusion simplex weights")

    locked = pd.read_csv(LOCKED_4277)
    if len(locked) != 4277 or locked["locked_row_id"].duplicated().any():
        raise RuntimeError("Expected 4,277 unique frozen locked predictions")
    fused = np.zeros(len(locked), dtype=np.float64)
    for member, weight in weights.items():
        if member not in locked:
            raise RuntimeError(f"Missing frozen member prediction: {member}")
        score_column = f"score__multiobjective__{member.removeprefix('pred__')}"
        locked[score_column] = rank01(locked[member].to_numpy(float))
        fused += weight * locked[score_column].to_numpy(float)
    locked["score__final_fusion_multiobjective"] = fused
    locked["D_rank_desc_multiobjective"] = locked["score__final_fusion_multiobjective"].rank(
        method="min", ascending=False
    ).astype(int)
    locked.to_csv(OUT_4277, index=False)

    cohort_ids = pd.read_csv(COHORT_4000, usecols=["locked_row_id"])
    if len(cohort_ids) != 4000 or cohort_ids["locked_row_id"].duplicated().any():
        raise RuntimeError("Expected the fixed 4,000-row evaluation cohort")
    evaluation = cohort_ids.merge(locked, on="locked_row_id", how="left", validate="one_to_one")
    if len(evaluation) != 4000 or evaluation["score__final_fusion_multiobjective"].isna().any():
        raise RuntimeError("Multiobjective evaluation coverage is incomplete")
    evaluation.to_csv(OUT_4000, index=False)

    y = evaluation["D"].to_numpy(float)
    rows = []
    for model, column in (
        ("final_fusion_spearman_primary_old", "score__final_fusion"),
        ("final_fusion_spearman_ndcg10_multiobjective", "score__final_fusion_multiobjective"),
    ):
        rows.append({"model": model, "prediction_column": column, "n": len(y), **ranking_metrics(y, evaluation[column].to_numpy(float))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_METRICS, index=False)

    manifest = {
        "status": "evaluated_after_fold0_fusion_freeze",
        "selection_data": "candidate100-clean fold-0 shared graph-valid subset only",
        "selection_rows": int(payload["n_shared_graph_valid"]),
        "selection_objectives": ["Spearman", "NDCG@10%"],
        "evaluation_data": "fixed chemistry-curated 4,000-row cohort",
        "evaluation_rows": len(evaluation),
        "rank_normalization_reference_rows": len(locked),
        "evaluation_labels_used_for_weight_selection": False,
        "weights_file": str(WEIGHTS),
        "weights_sha256": sha256(WEIGHTS),
        "predictions_sha256": sha256(OUT_4000),
        "weights_by_member": weights,
        "comparability": "Only the fusion selection rule and weights changed; full-data member models and evaluation cohort stayed fixed.",
        "caveat": "The 4,000-row cohort is chemistry-curated and is not the original untouched locked test.",
    }
    (FUSION_DIR / "evaluation4000_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
