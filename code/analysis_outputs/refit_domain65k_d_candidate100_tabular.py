#!/usr/bin/env python3
"""Refit the frozen candidate100-clean tabular D ensemble on all development rows."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import run_domain65k_d_v2_tabular_augmented as augmented
from run_domain65k_d_v2_tabular_candidate100_fold0 import frozen_fold0_features, load_candidate100_frame


OUT = augmented.train.RUN_DIR / "domain65k_d_candidate100_tabular_full"
ARTIFACTS = OUT / "model_artifacts"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame, _ = load_candidate100_frame()
    selected = frozen_fold0_features(frame)
    x = frame[selected].to_numpy(np.float32)
    y = frame["D"].to_numpy(float)
    members = []

    def fit(slot: str):
        estimator = augmented.model(slot, 0)
        estimator.fit(x, augmented.target_for(y, slot))
        if slot.startswith("lgbm"):
            filename, model_type = f"full_{slot}.txt", "lightgbm"
            estimator.booster_.save_model(str(ARTIFACTS / filename))
        else:
            filename, model_type = f"full_{slot}.cbm", "catboost"
            estimator.save_model(str(ARTIFACTS / filename))
        return slot, filename, model_type

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fit, slot) for slot in augmented.SLOTS]
        for future in as_completed(futures):
            slot, filename, model_type = future.result()
            members.append({"slot": slot, "model_type": model_type, "model_file": filename,
                            "feature_columns": selected})
            print(f"candidate100 full tabular completed slot={slot}", flush=True)

    manifest = {
        "status": "frozen_fold0_config_refit_on_all_development",
        "n_candidate100_clean": len(frame), "n_features": len(selected), "slots": list(augmented.SLOTS),
        "candidate_overlap_policy": "candidate100 canonical-SMILES and domain-index overlap excluded before fitting",
        "feature_policy": "exact frozen fold-0 feature manifest from prior 0.81 compatibility diagnostic",
        "locked_status": "not read", "members": sorted(members, key=lambda item: item["slot"]),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
