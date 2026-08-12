#!/usr/bin/env python3
"""Compute weighted, scale-aligned TreeSHAP for the frozen tabular component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis_outputs/paper_figures_ml_workflow/source_data"
INPUT = SOURCE / "feature_interpretability_input.pkl.gz"
TAB = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs/domain65k_d_candidate100_tabular_full"
XGB = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs/domain65k_d_candidate100_xgb_full"
WEIGHTS = ROOT / "analysis_outputs/qme14s_training/domain65k/model_runs/domain65k_d_candidate100_fold0_fusion_multiobjective/multiobjective_weights.json"


def feature_family(name: str) -> str:
    if name.startswith("atom3d__"):
        return "Atom-resolved 3D"
    if name.startswith("mc3__"):
        return "Three-conformer xTB"
    if name.startswith("single__"):
        return "Single-conformer xTB"
    if name.startswith("derived__"):
        return "Dipole-derived"
    return "2D descriptors"


def load_model_shap(member: dict, model_root: Path, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = [str(value) for value in member["feature_columns"]]
    values = frame[features].to_numpy(np.float32)
    source = model_root / "model_artifacts" / str(member["model_file"])
    model_type = str(member["model_type"])

    if model_type == "lightgbm":
        import lightgbm as lgb

        print({"stage": "load", "model": member["slot"]}, flush=True)
        model = lgb.Booster(model_file=str(source))
        print({"stage": "predict", "model": member["slot"]}, flush=True)
        prediction = np.asarray(model.predict(values, num_threads=4), dtype=np.float64)
        print({"stage": "contrib", "model": member["slot"]}, flush=True)
        contributions = np.asarray(
            model.predict(values, pred_contrib=True, num_threads=4),
            dtype=np.float64,
        )
    elif model_type == "catboost":
        import catboost as cb

        model = cb.CatBoostRegressor()
        model.load_model(str(source))
        pool = cb.Pool(values, feature_names=features)
        prediction = np.asarray(model.predict(pool), dtype=np.float64)
        contributions = np.asarray(model.get_feature_importance(pool, type="ShapValues"), dtype=np.float64)
    elif model_type == "xgboost":
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(str(source))
        matrix = xgb.DMatrix(values, feature_names=features)
        prediction = np.asarray(model.predict(matrix), dtype=np.float64)
        contributions = np.asarray(model.predict(matrix, pred_contribs=True), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    shap_values = contributions[:, :-1]
    residual = np.max(np.abs(contributions[:, -1] + shap_values.sum(axis=1) - prediction))
    if residual > 2e-3:
        raise RuntimeError(f"TreeSHAP additivity failure for {member['slot']}: {residual}")
    return prediction, shap_values


def configuration() -> tuple[dict, dict[str, tuple[dict, Path]], dict[str, float], dict[str, float]]:
    tab_manifest = json.loads((TAB / "manifest.json").read_text(encoding="utf-8"))
    xgb_manifest = json.loads((XGB / "manifest.json").read_text(encoding="utf-8"))
    weight_payload = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    weights = {str(key).removeprefix("pred__"): float(value) for key, value in weight_payload["weights_by_member"].items()}
    members = {
        str(member["slot"]): (member, TAB if member["model_type"] != "xgboost" else XGB)
        for member in tab_manifest["members"] + xgb_manifest["members"]
    }
    active = {slot: weight for slot, weight in weights.items() if slot != "vector_mu" and weight > 0}
    tabular_weight = float(sum(active.values()))
    normalized_weights = {slot: weight / tabular_weight for slot, weight in active.items()}
    return weight_payload, members, weights, normalized_weights


def compute_member(slot: str, start: int | None = None, end: int | None = None) -> None:
    data = pd.read_pickle(INPUT, compression="gzip")
    _, members, weights, normalized_weights = configuration()
    if slot not in normalized_weights:
        raise ValueError(f"Inactive or unknown member: {slot}")
    if start is not None or end is not None:
        if start is None or end is None or not (0 <= start < end <= len(data)):
            raise ValueError("Both --start and --end must define a valid row slice")
        data = data.iloc[start:end].reset_index(drop=True)
    member, model_root = members[slot]
    features = [str(value) for value in member["feature_columns"]]
    prediction, shap_values = load_model_shap(member, model_root, data)
    scale = float(np.std(prediction))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid output scale for {slot}")
    suffix = f"_{start:04d}_{end:04d}" if start is not None and end is not None else ""
    np.savez_compressed(
        SOURCE / f"feature_interpretability_member_{slot}{suffix}.npz",
        prediction=prediction.astype(np.float32),
        shap_values=shap_values.astype(np.float32),
        features=np.asarray(features, dtype=str),
        prediction_sd=np.asarray([scale], dtype=np.float64),
    )
    print({
        "member": slot,
        "final_weight": weights[slot],
        "within_tabular_weight": normalized_weights[slot],
        "prediction_sd": scale,
    }, flush=True)


def combine_member(slot: str) -> None:
    paths = sorted(SOURCE.glob(f"feature_interpretability_member_{slot}_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].npz"))
    if not paths:
        raise FileNotFoundError(f"No chunk attributions found for {slot}")
    predictions: list[np.ndarray] = []
    shap_blocks: list[np.ndarray] = []
    reference_features: np.ndarray | None = None
    for path in paths:
        payload = np.load(path)
        features = payload["features"].astype(str)
        scale = float(payload["prediction_sd"][0])
        if reference_features is None:
            reference_features = features
        elif not np.array_equal(features, reference_features):
            raise RuntimeError(f"Feature mismatch in {path.name}")
        predictions.append(payload["prediction"].astype(np.float32))
        shap_blocks.append(payload["shap_values"].astype(np.float32))
    prediction = np.concatenate(predictions)
    shap_values = np.concatenate(shap_blocks)
    if len(prediction) != 4000:
        raise RuntimeError(f"Expected 4,000 combined rows for {slot}, found {len(prediction)}")
    np.savez_compressed(
        SOURCE / f"feature_interpretability_member_{slot}.npz",
        prediction=prediction,
        shap_values=shap_values,
        features=reference_features,
        prediction_sd=np.asarray([float(np.std(prediction))], dtype=np.float64),
    )
    print({"combined_member": slot, "rows": len(prediction), "chunks": len(paths)}, flush=True)


def aggregate_members() -> None:
    data = pd.read_pickle(INPUT, compression="gzip")
    _, members, weights, normalized_weights = configuration()
    tabular_weight = float(sum(weights[slot] for slot in normalized_weights))

    all_features = sorted({str(name) for slot in normalized_weights for name in members[slot][0]["feature_columns"]})
    position = {name: index for index, name in enumerate(all_features)}
    aggregate = np.zeros((len(data), len(all_features)), dtype=np.float64)
    diagnostics: list[dict] = []
    member_importance_rows: list[pd.DataFrame] = []

    for slot, normalized_weight in normalized_weights.items():
        source = SOURCE / f"feature_interpretability_member_{slot}.npz"
        if not source.exists():
            raise FileNotFoundError(f"Missing member attribution: {source}")
        payload = np.load(source)
        features = payload["features"].astype(str).tolist()
        prediction = payload["prediction"].astype(np.float64)
        shap_values = payload["shap_values"].astype(np.float64)
        scale = float(payload["prediction_sd"][0])
        indices = np.asarray([position[name] for name in features], dtype=int)
        standardized_shap = shap_values / scale
        aggregate[:, indices] += normalized_weight * standardized_shap
        member_abs = np.mean(np.abs(standardized_shap), axis=0)
        member_importance_rows.append(pd.DataFrame({
            "member": slot,
            "feature": features,
            "family": [feature_family(name) for name in features],
            "mean_abs_standardized_shap": member_abs,
            "importance_fraction_within_member": member_abs / member_abs.sum(),
            "within_tabular_weight": normalized_weight,
        }))
        diagnostics.append({
            "member": slot,
            "final_fusion_weight": weights[slot],
            "within_tabular_weight": normalized_weight,
            "prediction_sd": scale,
            "n_features": len(features),
        })
        print({"aggregate_member": slot, "weight": normalized_weight, "prediction_sd": scale}, flush=True)

    importance = np.mean(np.abs(aggregate), axis=0)
    signed_mean = np.mean(aggregate, axis=0)
    ranking = np.argsort(importance)[::-1]
    global_table = pd.DataFrame({
        "feature": all_features,
        "family": [feature_family(name) for name in all_features],
        "mean_abs_weighted_shap": importance,
        "mean_weighted_shap": signed_mean,
    }).sort_values("mean_abs_weighted_shap", ascending=False).reset_index(drop=True)
    global_table["importance_fraction"] = global_table["mean_abs_weighted_shap"] / global_table["mean_abs_weighted_shap"].sum()
    global_table.to_csv(SOURCE / "feature_interpretability_global_importance.csv", index=False)
    pd.concat(member_importance_rows, ignore_index=True).to_csv(
        SOURCE / "feature_interpretability_member_global_importance.csv", index=False
    )

    top_indices = ranking[:15]
    rows: list[pd.DataFrame] = []
    for feature_index in top_indices:
        feature = all_features[feature_index]
        values = data[feature].astype(float)
        finite = np.isfinite(values.to_numpy())
        percentile = np.full(len(values), np.nan, dtype=float)
        if finite.any():
            percentile[finite] = pd.Series(values.to_numpy()[finite]).rank(method="average", pct=True).to_numpy(float)
        rows.append(pd.DataFrame({
            "domain_index": data["domain_index"].to_numpy(int),
            "feature": feature,
            "family": feature_family(feature),
            "feature_value": values.to_numpy(float),
            "feature_percentile": percentile,
            "weighted_standardized_shap": aggregate[:, feature_index],
        }))
    pd.concat(rows, ignore_index=True).to_csv(SOURCE / "feature_interpretability_top15_points.csv", index=False)
    np.savez_compressed(
        SOURCE / "feature_interpretability_weighted_shap.npz",
        values=aggregate.astype(np.float32),
        features=np.asarray(all_features, dtype=str),
        domain_index=data["domain_index"].to_numpy(int),
    )
    pd.DataFrame(diagnostics).to_csv(SOURCE / "feature_interpretability_member_diagnostics.csv", index=False)
    family = global_table.groupby("family", as_index=False)["mean_abs_weighted_shap"].sum()
    family["importance_fraction"] = family["mean_abs_weighted_shap"] / family["mean_abs_weighted_shap"].sum()
    family.sort_values("importance_fraction", ascending=False).to_csv(
        SOURCE / "feature_interpretability_family_importance.csv", index=False
    )
    (SOURCE / "feature_interpretability_shap_manifest.json").write_text(
        json.dumps({
            "cohort": "frozen chemistry-curated 4,000-record evaluation set",
            "n_records": len(data),
            "n_unique_domain_index": int(data["domain_index"].nunique()),
            "scope": "tabular component only; vector branch is not represented",
            "aggregation": "TreeSHAP divided by each member prediction SD, then averaged using final nonzero tabular fusion weights normalized to sum one",
            "active_members": diagnostics,
            "tabular_weight_in_full_fusion": tabular_weight,
            "labels_used": False,
        }, indent=2),
        encoding="utf-8",
    )
    print(global_table.head(15).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--combine-member")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    SOURCE.mkdir(parents=True, exist_ok=True)
    if args.member:
        compute_member(args.member, args.start, args.end)
    elif args.combine_member:
        combine_member(args.combine_member)
    elif args.aggregate:
        aggregate_members()
    else:
        raise SystemExit("Specify --member SLOT or --aggregate")


if __name__ == "__main__":
    main()
