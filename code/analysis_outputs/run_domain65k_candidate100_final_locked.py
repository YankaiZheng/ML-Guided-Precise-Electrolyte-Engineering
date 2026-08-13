#!/usr/bin/env python3
"""Evaluate the frozen D-P Compass models on the published Test-4000 cohort.

This public entry point intentionally reads the released final-prediction table
rather than any exploratory or superseded internal cohort.  It reports the
paper's final multi-objective eight-member D fusion and the final single-
LightGBM P model.  No training, feature generation, or refitting occurs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY / "data" / "test" / "test4000_predictions.csv"
DEFAULT_OUTPUT = REPOSITORY / "results" / "test4000_final_metrics.json"

FINAL_D_SCORE = "score__final_fusion_multiobjective"
FINAL_P_SCORE = "P_pred__single_lgbm_raw"


def spearman(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(y_true.rank(method="average").corr(y_pred.rank(method="average"), method="pearson"))


def ndcg_at_fraction(y_true: np.ndarray, score: np.ndarray, fraction: float = 0.10) -> float:
    """NDCG used for the manuscript ranking analysis (higher target is better)."""
    k = max(1, int(np.ceil(len(y_true) * fraction)))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    predicted = np.argsort(-score, kind="mergesort")[:k]
    ideal = np.argsort(-y_true, kind="mergesort")[:k]
    # The paper reports raw physical-property values as gains.  Shifting the
    # target changes P NDCG when its minimum is non-zero, so do not translate
    # the target before calculating DCG.
    gains = np.asarray(y_true, dtype=float)
    return float(np.dot(gains[predicted], discount) / np.dot(gains[ideal], discount))


def evaluate(y_true: pd.Series, score: pd.Series) -> dict[str, float]:
    values = y_true.to_numpy(dtype=float)
    predictions = score.to_numpy(dtype=float)
    return {
        "spearman": spearman(y_true, score),
        "ndcg_at_10pct": ndcg_at_fraction(values, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"D", "P", FINAL_D_SCORE, FINAL_P_SCORE}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required published columns: {sorted(missing)}")
    if len(frame) != 4000:
        raise ValueError(f"The final internal test cohort must contain 4,000 rows, found {len(frame)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Published test table contains missing targets or final predictions")

    report = {
        "cohort": "fixed_internal_test4000",
        "n_molecules": int(len(frame)),
        "D": {
            "model": "final_eight_member_rank_normalized_multiobjective_fusion",
            "prediction_column": FINAL_D_SCORE,
            **evaluate(frame["D"], frame[FINAL_D_SCORE]),
        },
        "P": {
            "model": "final_single_lightgbm",
            "prediction_column": FINAL_P_SCORE,
            **evaluate(frame["P"], frame[FINAL_P_SCORE]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
