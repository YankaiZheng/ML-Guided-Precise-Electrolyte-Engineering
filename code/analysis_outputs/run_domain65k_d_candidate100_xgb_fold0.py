#!/usr/bin/env python3
"""Candidate100-clean fold-0 XGBoost members using the historical frozen feature list."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import run_domain65k_d_v2_xgb as legacy
from run_domain65k_d_v2_tabular_candidate100_fold0 import load_candidate100_frame


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs"
OUT = RUN / "domain65k_d_candidate100_xgb_fold0"
ARTIFACTS = OUT / "model_artifacts"
FROZEN = RUN / "domain65k_d_v2_xgb/manifest.json"


def frozen_features(frame: pd.DataFrame) -> list[str]:
    members = [row for row in json.loads(FROZEN.read_text(encoding="utf-8"))["members"] if int(row["outer_fold"]) == 0]
    feature_sets = {tuple(row["feature_columns"]) for row in members}
    if len(feature_sets) != 1:
        raise RuntimeError("Historical fold-0 XGB slots do not share one frozen feature list")
    selected = list(feature_sets.pop())
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise RuntimeError(f"Frozen XGB features are missing: {missing[:10]}")
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame, _ = load_candidate100_frame()
    selected = frozen_features(frame)
    y = frame["D"].to_numpy(float)
    folds = frame["cv_fold"].to_numpy(int)
    train_idx, val_idx = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    x_train = frame.loc[train_idx, selected].to_numpy(np.float32)
    x_val = frame.loc[val_idx, selected].to_numpy(np.float32)
    output = frame.loc[val_idx, ["domain_index", "canonical_smiles", "D", "cv_fold"]].copy().reset_index(drop=True)
    metrics, members = [], []

    def fit(slot: str):
        estimator = legacy.model(slot, 0)
        estimator.fit(x_train, legacy.target_for(y[train_idx], slot))
        prediction = np.asarray(estimator.predict(x_val), dtype=float)
        filename = f"fold0_{slot}.json"
        estimator.save_model(str(ARTIFACTS / filename))
        return slot, prediction, filename

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fit, slot) for slot in legacy.SLOTS]
        for future in as_completed(futures):
            slot, prediction, filename = future.result()
            output[f"pred__{slot}"] = prediction
            metric = legacy.extended_metrics(y[val_idx], prediction, legacy.SEED)
            metrics.append({"fold": 0, "slot": slot, "n_features": len(selected), **metric})
            members.append({"slot": slot, "model_type": "xgboost", "model_file": filename,
                            "feature_columns": selected})
            print(f"candidate100 XGB fold=0 slot={slot} sp={metric['spearman']:.6f}", flush=True)
    output.to_csv(OUT / "fold0_predictions.csv", index=False)
    pd.DataFrame(metrics).sort_values("slot").to_csv(OUT / "fold0_metrics.csv", index=False)
    manifest = {
        "status": "candidate100_clean_frozen_fold0_config", "n_candidate100_clean": len(frame),
        "n_train": len(train_idx), "n_outer": len(val_idx), "n_features": len(selected),
        "feature_policy": "exact historical fold-0 XGB feature manifest", "members": members,
        "locked_status": "not read",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
