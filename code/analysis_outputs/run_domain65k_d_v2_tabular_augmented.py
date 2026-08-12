#!/usr/bin/env python3
"""Five-fold D tabular branch using legacy deployable features plus raw atom3d derivatives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

import run_domain65k_d_model_search as train


RUN = train.RUN_DIR
DERIVED = train.DOMAIN_DIR / "features_v2_atom3d_derived/chunks"
OUT = RUN / "domain65k_d_v2_tabular_augmented"
ARTIFACTS = OUT / "model_artifacts"
SLOTS = ("lgbm_raw", "lgbm_rank", "lgbm_normal", "cat_raw", "cat_rank")
SEED = 20260713
CHEAP_TOP_K = 512
ATOM3D_TOP_K = 128


def rank_target(y: np.ndarray) -> np.ndarray:
    return pd.Series(y).rank(method="average", pct=True).to_numpy(np.float32)


def target_for(y: np.ndarray, slot: str) -> np.ndarray:
    if slot.endswith("raw"):
        return y.astype(np.float32)
    ranks = rank_target(y)
    if slot.endswith("normal"):
        return norm.ppf(np.clip(ranks, 1e-4, 1 - 1e-4)).astype(np.float32)
    return ranks


def load_augmented() -> tuple[pd.DataFrame, list[str]]:
    frame, base_features = train.load_matrix(["cheap", "single", "mc3"])
    paths = sorted(DERIVED.glob("chunk_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No atom3d-derived chunks under {DERIVED}")
    derived = pd.concat([pd.read_pickle(path) for path in paths], ignore_index=True)
    derived = derived.sort_values("domain_index").drop_duplicates("domain_index", keep="first")
    if derived["domain_index"].duplicated().any():
        raise RuntimeError("Duplicate domain_index in atom3d-derived table")
    derived_cols = [
        col for col in derived.select_dtypes(include=[np.number]).columns
        if col != "domain_index" and not col.endswith("_status")
    ]
    if len(derived_cols) < 100:
        raise RuntimeError(f"Atom3d-derived table unexpectedly small: {len(derived_cols)} columns")
    aligned = frame[["domain_index"]].merge(
        derived[["domain_index", "status"] + derived_cols], on="domain_index", how="left", validate="many_to_one"
    )
    keep = aligned["status"].eq("ok").to_numpy()
    if not keep.all():
        print(f"atom3d-derived complete-case filter: kept={int(keep.sum())} dropped={int((~keep).sum())}", flush=True)
        frame = frame.loc[keep].reset_index(drop=True)
        aligned = aligned.loc[keep].reset_index(drop=True)
    aligned = aligned.drop(columns=["status", "domain_index"])
    aligned.columns = [str(col) for col in aligned.columns]
    frame = pd.concat([frame.reset_index(drop=True), aligned.reset_index(drop=True)], axis=1)
    features = base_features + derived_cols
    if len(features) != len(set(features)):
        duplicated = pd.Series(features).value_counts()
        raise RuntimeError(f"Duplicate augmented feature names: {duplicated[duplicated > 1].index.tolist()[:10]}")
    if len(frame) != 60672:
        raise RuntimeError(f"Expected 60672 candidate-excluded augmented rows, got {len(frame)}")
    return frame, features


def d_allowed(name: str) -> bool:
    low = name.lower()
    hard_removed = (
        "molmr", "mol_mr", "mol_polarizability", "mol_c6", "mol_c8",
        "alpha_per", "derived__c6", "derived__c8",
    )
    return not any(token in low for token in hard_removed)


def fold_features(frame: pd.DataFrame, features: list[str], train_idx: np.ndarray) -> list[str]:
    allowed = [name for name in features if d_allowed(name)]
    cheap = [name for name in allowed if name.startswith("cheap__")]
    atom3d = [name for name in allowed if name.startswith("atom3d__")]
    physics = [name for name in allowed if not name.startswith("cheap__") and not name.startswith("atom3d__")]
    y_rank = rank_target(frame.loc[train_idx, "D"].to_numpy(float)).astype(np.float64)
    y_rank -= y_rank.mean()

    def order(names: list[str]) -> np.ndarray:
        values = frame.loc[train_idx, names].to_numpy(np.float32)
        means = np.nanmean(values, axis=0)
        centered = np.where(np.isfinite(values), values - means, 0.0)
        denom = np.sqrt(np.sum(centered * centered, axis=0) * np.dot(y_rank, y_rank))
        correlation = np.divide(y_rank @ centered, denom, out=np.zeros_like(denom), where=denom > 0)
        return np.argsort(np.abs(correlation))[::-1]

    cheap_order = order(cheap)
    atom_order = order(atom3d)
    selected = (
        physics
        + [atom3d[i] for i in atom_order[:ATOM3D_TOP_K]]
        + [cheap[i] for i in cheap_order[:CHEAP_TOP_K]]
    )
    if not (1150 <= len(selected) <= 1300):
        raise RuntimeError(f"Unexpected augmented D feature count: {len(selected)}")
    return selected


def model(slot: str, fold: int):
    seed = SEED + fold * 100 + len(slot)
    if slot.startswith("lgbm"):
        return lgb.LGBMRegressor(
            objective="regression", n_estimators=1800, learning_rate=0.025,
            num_leaves=95, min_child_samples=20, subsample=0.92,
            colsample_bytree=0.82, reg_alpha=0.04, reg_lambda=0.16,
            max_bin=255, force_col_wise=True, random_state=seed,
            n_jobs=3, verbose=-1,
        )
    return cb.CatBoostRegressor(
        loss_function="RMSE", iterations=1600, learning_rate=0.035,
        depth=8, l2_leaf_reg=5.0, random_seed=seed,
        thread_count=3, verbose=False, allow_writing_files=False,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame, features = load_augmented()
    y = frame["D"].to_numpy(float)
    folds = frame["cv_fold"].to_numpy(int)
    oof = frame[["domain_index", "canonical_smiles", "D", "cv_fold"]].copy()
    metrics, members = [], []
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
            if slot.startswith("lgbm"):
                filename, model_type = f"fold{fold}_{slot}.txt", "lightgbm"
                estimator.booster_.save_model(str(ARTIFACTS / filename))
            else:
                filename, model_type = f"fold{fold}_{slot}.cbm", "catboost"
                estimator.save_model(str(ARTIFACTS / filename))
            return slot, pred, filename, model_type

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(fit, slot) for slot in SLOTS]
            for future in as_completed(futures):
                slot, pred, filename, model_type = future.result()
                oof.loc[val_idx, f"pred__{slot}"] = pred
                metric = train.evaluate(y[val_idx], pred, SEED + fold)
                metrics.append({"fold": fold, "slot": slot, "n_features": len(selected), **metric})
                members.append({"outer_fold": fold, "slot": slot, "model_type": model_type, "model_file": filename, "feature_columns": selected})
                print(f"D augmented fold={fold} slot={slot} sp={metric['spearman']:.6f}", flush=True)
        oof.to_pickle(OUT / "oof.pkl")
        (OUT / f"fold{fold}_feature_manifest.json").write_text(
            json.dumps({"fold": fold, "n_features": len(selected), "features": selected}, indent=2), encoding="utf-8"
        )
    pred_cols = [f"pred__{slot}" for slot in SLOTS]
    if any(oof[col].isna().any() for col in pred_cols):
        raise RuntimeError("Augmented D OOF is incomplete")
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    manifest = {
        "target": "D", "n_cv": len(frame), "slots": list(SLOTS),
        "n_base_features": len(features), "cheap_top_k": CHEAP_TOP_K, "atom3d_top_k": ATOM3D_TOP_K,
        "raw_atom3d_derived": True, "hard_removed": ["MolMR/mol_mr", "polarizability", "C6", "C8", "alpha/C6/C8 derived"],
        "candidate_overlap_policy": "47 exact candidate structures excluded before any development fitting.",
        "members": members, "locked_status": "not evaluated",
        "forbidden_inputs": "No DFT q+/q-/HOMO/LUMO/NPA/DFT xyz.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
