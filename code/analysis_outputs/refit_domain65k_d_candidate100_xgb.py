#!/usr/bin/env python3
"""Refit frozen candidate100-clean XGBoost members on all development rows."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import run_domain65k_d_v2_xgb as legacy
from run_domain65k_d_candidate100_xgb_fold0 import frozen_features
from run_domain65k_d_v2_tabular_candidate100_fold0 import load_candidate100_frame


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs/domain65k_d_candidate100_xgb_full"
ARTIFACTS = OUT / "model_artifacts"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame, _ = load_candidate100_frame()
    selected = frozen_features(frame)
    x = frame[selected].to_numpy(np.float32)
    y = frame["D"].to_numpy(float)
    members = []

    def fit(slot: str):
        estimator = legacy.model(slot, 0)
        estimator.fit(x, legacy.target_for(y, slot))
        filename = f"full_{slot}.json"
        estimator.save_model(str(ARTIFACTS / filename))
        return slot, filename

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fit, slot) for slot in legacy.SLOTS]
        for future in as_completed(futures):
            slot, filename = future.result()
            members.append({"slot": slot, "model_type": "xgboost", "model_file": filename,
                            "feature_columns": selected})
            print(f"candidate100 full XGB completed slot={slot}", flush=True)
    manifest = {
        "status": "frozen_fold0_config_refit_on_all_development", "n_candidate100_clean": len(frame),
        "n_features": len(selected), "members": sorted(members, key=lambda item: item["slot"]),
        "feature_policy": "exact historical fold-0 XGB feature manifest", "locked_status": "not read",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
