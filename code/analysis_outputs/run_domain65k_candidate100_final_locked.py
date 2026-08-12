#!/usr/bin/env python3
"""Locked inference for the frozen Candidate100-clean D/P models.

The prepare and predict phases never read D/P locked labels.  The finalize
phase is intentionally the sole code path that reads labels for one report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata

import run_domain65k_d_locked_test as locked_features
import run_domain65k_d_model_search as domain
from run_domain65k_p_candidate100_single_lgbm import ranking_metrics


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "analysis_outputs/qme14s_training/domain65k"
RUN = DOMAIN / "model_runs"
LOCKED = DOMAIN / "domain65k_locked_test_complete_features.csv"
V2 = ROOT / "analysis_outputs/candidate_curation_v2/domain65k_candidate_exact_exclusion.csv"
V3 = ROOT / "analysis_outputs/candidate_curation_v3/domain65k_candidate100_exact_exclusion.csv"
DERIVED = DOMAIN / "features_v2_atom3d_derived/chunks"
OUT = RUN / "domain65k_candidate100_final_locked"
UNLABELED = OUT / "candidate100_clean_locked_unlabeled.csv"
BASE_PREDICTIONS = OUT / "candidate100_clean_locked_base_predictions.csv"
VECTOR_PREDICTIONS = OUT / "candidate100_clean_locked_vector_predictions.csv"
FINAL_PREDICTIONS = OUT / "candidate100_clean_locked_predictions.csv"

TABULAR = RUN / "domain65k_d_candidate100_tabular_full"
XGB = RUN / "domain65k_d_candidate100_xgb_full"
P_SINGLE = RUN / "domain65k_p_candidate100_single_lgbm"
WEIGHTS = RUN / "domain65k_d_candidate100_fold0_fusion/manual_refined_weights_with_xgb.json"


def candidate_clean_locked(*, include_labels: bool) -> pd.DataFrame:
    cols = ["domain_index", "canonical_smiles", "heavy_atoms", "mol_wt"]
    if include_labels:
        cols += ["D", "P"]
    locked = pd.read_csv(LOCKED, usecols=cols, low_memory=False)
    exclusions = pd.concat([
        pd.read_csv(V2, usecols=["domain_index", "canonical_smiles"]),
        pd.read_csv(V3, usecols=["domain_index", "canonical_smiles"]),
    ], ignore_index=True).drop_duplicates()
    ids = set(exclusions["domain_index"].astype(int))
    smiles = set(exclusions["canonical_smiles"].astype(str))
    keep = ~locked["domain_index"].astype(int).isin(ids)
    keep &= ~locked["canonical_smiles"].astype(str).isin(smiles)
    locked = locked.loc[keep].sort_values("domain_index").reset_index(drop=True)
    if len(locked) != 4277:
        raise RuntimeError(f"Expected 4277 Candidate100-clean locked rows, found {len(locked)}")
    if locked["domain_index"].astype(int).isin(ids).any() or locked["canonical_smiles"].astype(str).isin(smiles).any():
        raise RuntimeError("Candidate overlap leaked into locked inference")
    locked.insert(0, "locked_row_id", np.arange(len(locked), dtype=int))
    return locked


def rank01(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average").astype(float) / len(values)


def load_atom3d_derived(wanted: np.ndarray) -> pd.DataFrame:
    blocks = [pd.read_pickle(path) for path in sorted(DERIVED.glob("chunk_*.pkl"))]
    all_rows = pd.concat(blocks, ignore_index=True).set_index("domain_index")
    result = all_rows.loc[wanted].reset_index()
    if not result["status"].eq("ok").all():
        raise RuntimeError("Candidate-clean locked rows contain incomplete atom3d-derived records")
    return result.drop(columns=["status"])


def build_feature_matrices(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, _ = locked_features.build_feature_frame(meta, ["cheap", "single", "mc3"])
    atom = load_atom3d_derived(meta["domain_index"].to_numpy(int))
    atom = atom.drop(columns=["domain_index"])
    augmented = pd.concat([base.reset_index(drop=True), atom.reset_index(drop=True)], axis=1)
    return base, augmented


def predict_base() -> None:
    if not UNLABELED.exists():
        raise FileNotFoundError("Run --phase prepare before --phase predict-base")
    meta = pd.read_csv(UNLABELED)
    base, augmented = build_feature_matrices(meta)
    tab_manifest = json.loads((TABULAR / "manifest.json").read_text())
    xgb_manifest = json.loads((XGB / "manifest.json").read_text())
    result = meta[["locked_row_id", "domain_index", "canonical_smiles"]].copy()

    for member in tab_manifest["members"]:
        slot = str(member["slot"])
        features = [str(value) for value in member["feature_columns"]]
        if any(name not in augmented for name in features):
            raise RuntimeError(f"Missing tabular feature for {slot}")
        source = TABULAR / "model_artifacts" / member["model_file"]
        x = augmented[features].astype(np.float32)
        if member["model_type"] == "lightgbm":
            prediction = lgb.Booster(model_file=str(source)).predict(x)
        else:
            model = cb.CatBoostRegressor()
            model.load_model(str(source))
            prediction = model.predict(x)
        result[f"pred__{slot}"] = np.asarray(prediction, dtype=float)

    for member in xgb_manifest["members"]:
        slot = str(member["slot"])
        features = [str(value) for value in member["feature_columns"]]
        if any(name not in base for name in features):
            raise RuntimeError(f"Missing XGBoost feature for {slot}")
        model = xgb.XGBRegressor()
        model.load_model(str(XGB / "model_artifacts" / member["model_file"]))
        result[f"pred__{slot}"] = np.asarray(model.predict(base[features].to_numpy(np.float32)), dtype=float)

    p_features = pd.read_csv(P_SINGLE / "fold0_feature_manifest.csv")["feature"].astype(str).tolist()
    if any(name not in base for name in p_features):
        raise RuntimeError("Missing frozen P feature")
    p_model = lgb.Booster(model_file=str(P_SINGLE / "model_artifacts/full_single_raw.txt"))
    result["P_pred__single_lgbm_raw"] = p_model.predict(base[p_features].astype(np.float32))
    if result.filter(regex=r"^(pred__|P_pred__)").isna().any().any():
        raise RuntimeError("Base D/P prediction contains NaN")
    result.to_csv(BASE_PREDICTIONS, index=False)
    print({"n_predicted": len(result), "output": str(BASE_PREDICTIONS)}, flush=True)


def finalize() -> None:
    if not BASE_PREDICTIONS.exists() or not VECTOR_PREDICTIONS.exists():
        raise FileNotFoundError("Base and vector predictions must exist before --phase finalize")
    # The sole locked-label read in this workflow occurs here, after model and
    # fusion configuration are frozen and all predictions are materialized.
    labels = candidate_clean_locked(include_labels=True)
    base = pd.read_csv(BASE_PREDICTIONS)
    vector = pd.read_csv(VECTOR_PREDICTIONS)
    merged = labels.merge(base, on="locked_row_id", how="inner", validate="one_to_one", suffixes=("", "_base"))
    merged = merged.merge(
        vector[["locked_row_id", "pred__vector_mu"]],
        on="locked_row_id", how="inner", validate="one_to_one",
    )
    if len(merged) != 4277:
        raise RuntimeError("Final locked prediction coverage is incomplete")
    payload = json.loads(WEIGHTS.read_text())
    weights = {name.removeprefix("pred__"): float(value) for name, value in payload["weights_by_member"].items()}
    fused = np.zeros(len(merged), dtype=float)
    for slot, weight in weights.items():
        column = f"pred__{slot}"
        if column not in merged:
            raise RuntimeError(f"Frozen fusion member missing: {column}")
        merged[f"score__{slot}"] = rank01(merged[column].to_numpy(float))
        fused += weight * merged[f"score__{slot}"].to_numpy(float)
    merged["score__final_fusion"] = fused
    merged["D_rank_desc"] = merged["score__final_fusion"].rank(method="min", ascending=False).astype(int)
    merged["P_rank_desc"] = merged["P_pred__single_lgbm_raw"].rank(method="min", ascending=False).astype(int)
    merged.to_csv(FINAL_PREDICTIONS, index=False)
    metrics = pd.DataFrame([
        {"target": "D", "model": "candidate100_final_eight_member", **ranking_metrics(merged["D"].to_numpy(float), fused)},
        {"target": "P", "model": "candidate100_single_lgbm_raw", **ranking_metrics(merged["P"].to_numpy(float), merged["P_pred__single_lgbm_raw"].to_numpy(float))},
    ])
    metrics.to_csv(OUT / "candidate100_clean_locked_metrics.csv", index=False)
    (OUT / "final_manifest.json").write_text(json.dumps({
        "n_locked_before_candidate_audit": 4294,
        "candidate_overlap_removed": 17,
        "n_candidate100_clean_locked": 4277,
        "d_fusion_weights": weights,
        "p_model": "single_lightgbm_raw_512_features",
        "locked_labels_read_after_frozen_predictions": True,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "predict-base", "finalize"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        candidate_clean_locked(include_labels=False).to_csv(UNLABELED, index=False)
        print({"n_candidate100_clean_locked": 4277, "output": str(UNLABELED)}, flush=True)
    elif args.phase == "predict-base":
        predict_base()
    else:
        finalize()


if __name__ == "__main__":
    main()
