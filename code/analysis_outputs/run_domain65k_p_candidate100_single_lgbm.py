#!/usr/bin/env python3
"""Candidate100-clean single-LightGBM P model with frozen fold-0 selection.

The model starts from the complete deployable cheap/single-xTB/MC3-xTB feature
pool.  Feature selection is performed only on fold-0's training partition;
the resulting list is then frozen for the one-shot full-development refit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import run_domain65k_d_model_search as domain


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs"
OUT = RUN / "domain65k_p_candidate100_single_lgbm"
ARTIFACTS = OUT / "model_artifacts"
EXCLUSION = ROOT / "analysis_outputs/candidate_curation_v3/domain65k_candidate100_exact_exclusion.csv"

FOLD = 0
SEED = 20260714
N_FEATURES = 512


def rank01(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average").astype(np.float64) / len(values)


def load_candidate100_clean() -> tuple[pd.DataFrame, list[str]]:
    frame, features = domain.load_matrix(["cheap", "single", "mc3"])
    excluded = pd.read_csv(EXCLUSION, usecols=["domain_index", "canonical_smiles"])
    excluded_ids = set(excluded["domain_index"].astype(int))
    excluded_smiles = set(excluded["canonical_smiles"].astype(str))
    keep = ~frame["domain_index"].astype(int).isin(excluded_ids)
    keep &= ~frame["canonical_smiles"].astype(str).isin(excluded_smiles)
    frame = frame.loc[keep].reset_index(drop=True)
    if len(frame) != 60641:
        raise RuntimeError(f"Expected 60641 Candidate100-clean rows, found {len(frame)}")
    if frame["canonical_smiles"].astype(str).isin(excluded_smiles).any():
        raise RuntimeError("Candidate100 canonical-SMILES leakage in P development frame")
    return frame, features


def select_features(frame: pd.DataFrame, features: list[str], train_idx: np.ndarray) -> list[str]:
    """Rank features on fold-0 train rows without touching the validation fold."""
    y_rank = rank01(frame.loc[train_idx, "P"].to_numpy(float))
    target = y_rank - y_rank.mean()
    target_norm = float(np.dot(target, target))
    association = np.zeros(len(features), dtype=np.float64)
    # Chunking avoids materializing several 48k x 3k dense arrays at once.
    for start in range(0, len(features), 64):
        names = features[start:start + 64]
        values = frame.loc[train_idx, names].to_numpy(np.float32)
        finite = np.isfinite(values)
        counts = finite.sum(axis=0)
        means = np.divide(
            np.where(finite, values, 0.0).sum(axis=0),
            counts,
            out=np.zeros(len(names), dtype=np.float32),
            where=counts > 0,
        )
        centered = np.where(finite, values - means, 0.0)
        denom = np.sqrt(np.sum(centered * centered, axis=0) * target_norm)
        association[start:start + len(names)] = np.divide(
            target @ centered,
            denom,
            out=np.zeros(len(names), dtype=np.float64),
            where=denom > 0,
        )
    order = np.argsort(np.abs(association))[::-1]
    selected = [features[index] for index in order[:N_FEATURES]]
    if len(selected) != N_FEATURES or len(selected) != len(set(selected)):
        raise RuntimeError("P feature selection did not produce 512 unique features")
    return selected


def model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1400,
        learning_rate=0.025,
        num_leaves=95,
        min_child_samples=20,
        subsample=0.92,
        colsample_bytree=0.84,
        reg_alpha=0.04,
        reg_lambda=0.16,
        max_bin=255,
        force_col_wise=True,
        random_state=SEED,
        n_jobs=8,
        verbose=-1,
    )


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(score)[::-1]
    ideal = np.argsort(y)[::-1]
    result: dict[str, float] = {
        "spearman": float(spearmanr(y, score).statistic),
        "kendall_tau": float(kendalltau(y, score).statistic),
        "pairwise_accuracy": float(domain.evaluate(y, score, SEED)["pairwise_accuracy"]),
    }
    for fraction, name in ((0.05, "05pct"), (0.10, "10pct"), (0.20, "20pct")):
        k = max(1, int(np.ceil(len(y) * fraction)))
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float((y[order[:k]] * discount).sum())
        idcg = float((y[ideal[:k]] * discount).sum())
        overlap = float(len(set(order[:k]) & set(ideal[:k])) / k)
        result[f"ndcg_at_{name}"] = dcg / idcg if idcg else float("nan")
        result[f"top_overlap_at_{name}"] = overlap
        result[f"ef_at_{name}"] = overlap / fraction
    calibrated = LinearRegression().fit(score.reshape(-1, 1), y).predict(score.reshape(-1, 1))
    result.update({
        "pcc": float(pearsonr(y, calibrated).statistic),
        "r2": float(r2_score(y, calibrated)),
        "mae_calibrated": float(mean_absolute_error(y, calibrated)),
        "rmse": float(mean_squared_error(y, calibrated) ** 0.5),
    })
    return result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_fold0() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame, features = load_candidate100_clean()
    folds = frame["cv_fold"].to_numpy(int)
    train_idx = np.flatnonzero(folds != FOLD)
    val_idx = np.flatnonzero(folds == FOLD)
    if len(train_idx) != 48511 or len(val_idx) != 12130:
        raise RuntimeError(f"Unexpected fold-0 split: train={len(train_idx)}, validation={len(val_idx)}")
    selected = select_features(frame, features, train_idx)
    estimator = model()
    estimator.fit(
        frame.loc[train_idx, selected].astype(np.float32),
        frame.loc[train_idx, "P"].to_numpy(np.float32),
    )
    prediction = np.asarray(estimator.predict(frame.loc[val_idx, selected].astype(np.float32)), dtype=float)
    estimator.booster_.save_model(str(ARTIFACTS / "fold0_single_raw.txt"))
    metrics = ranking_metrics(frame.loc[val_idx, "P"].to_numpy(float), prediction)
    frame.loc[val_idx, ["domain_index", "canonical_smiles", "P", "cv_fold"]].assign(
        pred__single_lgbm_raw=prediction
    ).to_csv(OUT / "fold0_predictions.csv", index=False)
    pd.Series(selected, name="feature").to_csv(OUT / "fold0_feature_manifest.csv", index=False)
    pd.DataFrame([{"fold": FOLD, "n_features": len(selected), **metrics}]).to_csv(OUT / "fold0_metrics.csv", index=False)
    write_json(OUT / "fold0_manifest.json", {
        "target": "P",
        "model": "single_lightgbm_raw",
        "n_candidate100_clean": len(frame),
        "fold": FOLD,
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_features_before_selection": len(features),
        "n_features_after_selection": len(selected),
        "feature_selection": "top 512 absolute train-fold association with P percentile rank",
        "candidate_overlap_policy": "Candidate100 domain-index and canonical-SMILES overlap excluded before selection and fitting",
        "locked_status": "not read",
        "metrics": metrics,
    })
    print(json.dumps(metrics, indent=2), flush=True)


def run_full() -> None:
    feature_path = OUT / "fold0_feature_manifest.csv"
    if not feature_path.exists():
        raise FileNotFoundError("Run --phase fold0 before --phase full to freeze the feature list")
    frame, _ = load_candidate100_clean()
    selected = pd.read_csv(feature_path)["feature"].astype(str).tolist()
    if len(selected) != N_FEATURES or any(name not in frame for name in selected):
        raise RuntimeError("Frozen fold-0 P feature list is invalid for full refit")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    estimator = model()
    estimator.fit(frame[selected].astype(np.float32), frame["P"].to_numpy(np.float32))
    estimator.booster_.save_model(str(ARTIFACTS / "full_single_raw.txt"))
    write_json(OUT / "full_manifest.json", {
        "status": "frozen_fold0_config_refit_on_all_development",
        "target": "P",
        "model": "single_lightgbm_raw",
        "n_candidate100_clean": len(frame),
        "n_features": len(selected),
        "feature_manifest": str(feature_path),
        "candidate_overlap_policy": "Candidate100 domain-index and canonical-SMILES overlap excluded before full fitting",
        "locked_status": "not read",
    })
    print("P full single-LightGBM refit completed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fold0", "full"), required=True)
    args = parser.parse_args()
    if args.phase == "fold0":
        run_fold0()
    else:
        run_full()


if __name__ == "__main__":
    main()
