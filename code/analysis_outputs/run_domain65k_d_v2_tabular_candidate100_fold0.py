#!/usr/bin/env python3
"""Candidate100-clean fold-0 tabular D branch for a fixed vector fusion check."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_domain65k_d_model_search as train
import run_domain65k_d_v2_tabular_augmented as augmented


ROOT = Path(__file__).resolve().parents[1]
OUT = train.RUN_DIR / "domain65k_d_v2_tabular_candidate100_fold0_frozen081"
ARTIFACTS = OUT / "model_artifacts"
EXCLUSION = ROOT / "analysis_outputs/candidate_curation_v3/domain65k_candidate100_exact_exclusion.csv"
FROZEN_FEATURE_MANIFEST = train.RUN_DIR / "domain65k_d_v2_tabular_augmented/fold0_feature_manifest.json"
FOLD = 0


def load_candidate100_frame() -> tuple[pd.DataFrame, list[str]]:
    frame, features = augmented.load_augmented()
    excluded = pd.read_csv(EXCLUSION, usecols=["domain_index", "canonical_smiles"])
    excluded_index = set(excluded["domain_index"].astype(int))
    excluded_smiles = set(excluded["canonical_smiles"].astype(str))
    keep = ~frame["domain_index"].astype(int).isin(excluded_index)
    keep &= ~frame["canonical_smiles"].astype(str).isin(excluded_smiles)
    frame = frame.loc[keep].reset_index(drop=True)
    if len(frame) != 60641:
        raise RuntimeError(f"Expected 60641 candidate100-clean rows, got {len(frame)}")
    if frame["canonical_smiles"].astype(str).isin(excluded_smiles).any():
        raise RuntimeError("Candidate canonical-SMILES leakage in tabular frame")
    return frame, features


def frozen_fold0_features(frame: pd.DataFrame) -> list[str]:
    payload = json.loads(FROZEN_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    selected = [str(name) for name in payload["features"]]
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise RuntimeError(f"Frozen fold-0 feature manifest has missing columns: {missing[:10]}")
    if len(selected) != 1221:
        raise RuntimeError(f"Expected 1221 frozen fold-0 features, got {len(selected)}")
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame, features = load_candidate100_frame()
    y = frame["D"].to_numpy(float)
    folds = frame["cv_fold"].to_numpy(int)
    train_idx = np.flatnonzero(folds != FOLD)
    val_idx = np.flatnonzero(folds == FOLD)
    if len(train_idx) != 48511 or len(val_idx) != 12130:
        raise RuntimeError(f"Unexpected candidate100 fold-0 split: train={len(train_idx)} val={len(val_idx)}")
    selected = frozen_fold0_features(frame)
    x_train = frame.loc[train_idx, selected].to_numpy(np.float32)
    x_val = frame.loc[val_idx, selected].to_numpy(np.float32)
    pred = frame.loc[val_idx, ["domain_index", "canonical_smiles", "D", "cv_fold"]].copy().reset_index(drop=True)
    metrics, members = [], []

    def fit(slot: str):
        estimator = augmented.model(slot, FOLD)
        estimator.fit(x_train, augmented.target_for(y[train_idx], slot))
        values = np.asarray(estimator.predict(x_val), dtype=float)
        if slot.startswith("lgbm"):
            filename, model_type = f"fold0_{slot}.txt", "lightgbm"
            estimator.booster_.save_model(str(ARTIFACTS / filename))
        else:
            filename, model_type = f"fold0_{slot}.cbm", "catboost"
            estimator.save_model(str(ARTIFACTS / filename))
        return slot, values, filename, model_type

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fit, slot) for slot in augmented.SLOTS]
        for future in as_completed(futures):
            slot, values, filename, model_type = future.result()
            pred[f"pred__{slot}"] = values
            metric = train.evaluate(y[val_idx], values, augmented.SEED + FOLD)
            metrics.append({"fold": FOLD, "slot": slot, "n_features": len(selected), **metric})
            members.append({"outer_fold": FOLD, "slot": slot, "model_type": model_type,
                            "model_file": filename, "feature_columns": selected})
            print(f"candidate100 tabular fold=0 slot={slot} sp={metric['spearman']:.6f}", flush=True)

    if pred.filter(like="pred__").isna().any().any():
        raise RuntimeError("Candidate100 tabular fold-0 predictions are incomplete")
    pred.to_csv(OUT / "fold0_predictions.csv", index=False)
    pd.DataFrame(metrics).sort_values("slot").to_csv(OUT / "fold0_metrics.csv", index=False)
    (OUT / "fold0_feature_manifest.json").write_text(
        json.dumps({"fold": FOLD, "n_features": len(selected), "features": selected}, indent=2), encoding="utf-8"
    )
    manifest = {
        "target": "D", "n_candidate100_clean": len(frame), "fold": FOLD,
        "n_train": len(train_idx), "n_outer": len(val_idx), "slots": list(augmented.SLOTS),
        "feature_policy": "exact frozen fold-0 feature manifest from prior 0.81 compatibility diagnostic",
        "candidate_overlap_policy": "candidate100 canonical-SMILES and domain-index overlap excluded before fitting",
        "candidate_smiles_leakage": False, "locked_status": "not read", "members": members,
        "forbidden_inputs": "No DFT q+/q-/HOMO/LUMO/NPA/DFT xyz.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(metrics).sort_values("slot").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
