#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import run_domain65k_d_model_search as train


RUN = train.RUN_DIR
OUT = RUN / "domain65k_d_v2_xgb"
ARTIFACTS = OUT / "model_artifacts"
SLOTS = ("xgb_raw", "xgb_rank")
SEED = 20260713
CHEAP_TOP_K = 512


def rank_target(y: np.ndarray) -> np.ndarray:
    return pd.Series(y).rank(method="average", pct=True).to_numpy(np.float32)


def target_for(y: np.ndarray, slot: str) -> np.ndarray:
    if slot.endswith("raw"):
        return y.astype(np.float32)
    ranks = rank_target(y)
    if slot.endswith("normal"):
        return norm.ppf(np.clip(ranks, 1e-4, 1 - 1e-4)).astype(np.float32)
    return ranks


def d_allowed(name: str) -> bool:
    low = name.lower()
    exact_noise = ("rd2d_molmr", "v3__mol_mr")
    response_only = (
        "mol_polarizability", "mol_c6", "mol_c8", "alpha_per_",
        "derived__c6", "derived__c8",
    )
    return not any(token in low for token in exact_noise + response_only)


def extended_metrics(y: np.ndarray, score: np.ndarray, seed: int) -> dict[str, float]:
    base = train.evaluate(y, score, seed)
    calibrated = LinearRegression().fit(score.reshape(-1, 1), y).predict(score.reshape(-1, 1))
    return {
        **base,
        "pcc": float(pearsonr(y, calibrated).statistic),
        "r2": float(r2_score(y, calibrated)),
        "rmse": float(mean_squared_error(y, calibrated) ** 0.5),
    }


def fold_features(frame: pd.DataFrame, features: list[str], train_idx: np.ndarray) -> list[str]:
    allowed = [name for name in features if d_allowed(name)]
    physics = [name for name in allowed if not name.startswith("cheap__")]
    cheap = [name for name in allowed if name.startswith("cheap__")]
    y_rank = rank_target(frame.loc[train_idx, "D"].to_numpy(float)).astype(np.float64)
    y_rank -= y_rank.mean()
    values = frame.loc[train_idx, cheap].to_numpy(np.float32)
    means = np.nanmean(values, axis=0)
    centered = values - means
    centered = np.where(np.isfinite(centered), centered, 0.0)
    denom = np.sqrt(np.sum(centered * centered, axis=0) * np.dot(y_rank, y_rank))
    correlation = np.divide(y_rank @ centered, denom, out=np.zeros_like(denom), where=denom > 0)
    order = np.argsort(np.abs(correlation))[::-1]
    selected_cheap = [cheap[i] for i in order[:CHEAP_TOP_K]]
    selected = physics + selected_cheap
    if not (900 <= len(selected) <= 1300):
        raise RuntimeError(f"Unexpected D tabular feature count: {len(selected)}")
    return selected


def model(slot: str, fold: int):
    seed = SEED + fold * 100 + len(slot)
    return xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=1800, learning_rate=0.025,
        max_depth=8, min_child_weight=3.0, subsample=0.90,
        colsample_bytree=0.85, reg_alpha=0.04, reg_lambda=0.20,
        tree_method="hist", max_bin=256, random_state=seed,
        n_jobs=4, verbosity=0,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame, features = train.load_matrix(["cheap", "single", "mc3"])
    if len(frame) != 60672:
        raise RuntimeError(f"Expected 60672 candidate-excluded rows, got {len(frame)}")
    y = frame["D"].to_numpy(float)
    folds = frame["cv_fold"].to_numpy(int)
    oof = frame[["domain_index", "canonical_smiles", "D", "cv_fold"]].copy()
    metrics = []
    members = []
    for fold in range(5):
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        selected = fold_features(frame, features, train_idx)
        x_train = frame.loc[train_idx, selected].to_numpy(np.float32)
        x_val = frame.loc[val_idx, selected].to_numpy(np.float32)
        def fit(slot: str):
            estimator = model(slot, fold)
            estimator.fit(x_train, target_for(y[train_idx], slot))
            pred = np.asarray(estimator.predict(x_val), dtype=float)
            name = f"fold{fold}_{slot}.json"
            estimator.save_model(str(ARTIFACTS / name))
            model_type = "xgboost"
            return slot, pred, name, model_type
        # Two four-thread jobs keep the 8-core Mac occupied.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(fit, slot) for slot in SLOTS]
            for future in as_completed(futures):
                slot, pred, filename, model_type = future.result()
                oof.loc[val_idx, f"pred__{slot}"] = pred
                result = extended_metrics(y[val_idx], pred, SEED + fold)
                metrics.append({"fold": fold, "slot": slot, "n_features": len(selected), **result})
                members.append({
                    "outer_fold": fold, "slot": slot, "model_type": model_type,
                    "model_file": filename, "feature_columns": selected,
                })
                print(f"D tabular fold={fold} slot={slot} sp={result['spearman']:.6f}", flush=True)
        oof.to_pickle(OUT / "oof.pkl")
        write = {"fold": fold, "n_features": len(selected), "features": selected}
        (OUT / f"fold{fold}_feature_manifest.json").write_text(json.dumps(write, indent=2), encoding="utf-8")
    if any(oof[f"pred__{slot}"].isna().any() for slot in SLOTS):
        raise RuntimeError("D tabular OOF is incomplete")
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    oof.to_pickle(OUT / "oof.pkl")
    manifest = {
        "target": "D", "n_cv": len(frame), "slots": list(SLOTS),
        "hard_removed": ["MolMR/mol_mr", "polarizability", "C6", "C8", "alpha/C6/C8 derived"],
        "cheap_selection": "Top 512 by abs Spearman on each outer-training partition only",
        "candidate_overlap_policy": "47 exact candidate structures excluded before any development fitting.",
        "members": members, "locked_status": "not evaluated",
        "forbidden_inputs": "No DFT q+/q-/HOMO/LUMO/NPA/DFT xyz.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
