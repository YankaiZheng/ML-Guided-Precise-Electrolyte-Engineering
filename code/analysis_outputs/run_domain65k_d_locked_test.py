#!/usr/bin/env python3
"""One-shot locked-test evaluation for the frozen 65k D ranking protocol.

The script intentionally refuses to run unless all five outer-fold OOF members
and the OOF-selected fusion weights already exist.  It never uses locked labels
for feature selection, parameter selection, iteration selection, or fusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import run_domain65k_d_model_search as train


DOMAIN = train.DOMAIN_DIR
RUN_DIR = train.RUN_DIR
LOCKED = DOMAIN / "domain65k_locked_test_complete_features.csv"


def build_feature_frame(split: pd.DataFrame, blocks: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Recreate the deployable matrix with the exact training feature names."""
    domain = pd.read_pickle(train.CHEAP).set_index("domain_index")
    wanted = split["domain_index"].to_numpy(int)
    if not set(wanted).issubset(domain.index):
        raise ValueError("Locked rows missing from cheap feature table.")

    parts: list[pd.DataFrame] = []
    excluded = {"domain_index", "D", "P", "cv_fold"}
    if "cheap" in blocks:
        parts.append(
            train.numeric_block(
                domain.loc[wanted].reset_index(), "cheap__", excluded, fill_failure=False
            )
        )
    if "single" in blocks:
        single = pd.read_pickle(train.SINGLE).set_index("domain_index")
        aligned = single.loc[wanted].reset_index()
        if not aligned["status"].eq("ok").all():
            raise ValueError("Locked test contains a failed single-conformer xTB row.")
        parts.append(
            train.numeric_block(
                aligned, "single__", excluded | {"xtb_reused_from_16k"}, fill_failure=False
            )
        )
    if "mc3" in blocks:
        mc3 = pd.read_pickle(train.MC3).set_index("domain_index")
        aligned = mc3.loc[wanted].reset_index()
        if not (
            aligned["status"].eq("ok")
            & aligned["xtb_mc3_n_conformer_success"].eq(3)
            & aligned["n_conformer_failure"].eq(0)
        ).all():
            raise ValueError("Locked test contains an incomplete multi-conformer xTB row.")
        parts.append(
            train.numeric_block(
                aligned, "mc3__", excluded | {"xtb_reused_from_16k"}, fill_failure=False
            )
        )

    x = pd.concat([part.reset_index(drop=True) for part in parts], axis=1)
    heavy = split["heavy_atoms"].to_numpy(np.float32)
    mw = split["mol_wt"].to_numpy(np.float32)
    if "single" in blocks:
        def col(name: str) -> np.ndarray:
            key = f"single__{name}"
            return x[key].to_numpy(np.float32) if key in x else np.full(len(x), np.nan, dtype=np.float32)

        mu, qmu = col("xtb_full_dipole_debye"), col("xtb_qonly_dipole_debye")
        alpha, c6, c8 = col("xtb_mol_polarizability_au"), col("xtb_mol_c6"), col("xtb_mol_c8")
        valid_mu = np.isfinite(mu)
        derived = pd.DataFrame(index=np.arange(len(x)))
        derived["derived__mu_per_heavy"] = np.where(valid_mu, mu / np.maximum(heavy, 1), np.nan)
        derived["derived__mu_per_mw"] = np.where(valid_mu, mu / np.maximum(mw, 1), np.nan)
        derived["derived__mu_minus_qmu_abs"] = np.where(valid_mu & np.isfinite(qmu), np.abs(mu - qmu), np.nan)
        derived["derived__qmu_over_mu"] = np.where(valid_mu & np.isfinite(qmu), qmu / np.maximum(np.abs(mu), 1e-5), np.nan)
        derived["derived__alpha_per_heavy"] = np.where(np.isfinite(alpha), alpha / np.maximum(heavy, 1), np.nan)
        derived["derived__alpha_per_mw"] = np.where(np.isfinite(alpha), alpha / np.maximum(mw, 1), np.nan)
        derived["derived__c6_per_heavy"] = np.where(np.isfinite(c6), c6 / np.maximum(heavy, 1), np.nan)
        derived["derived__c8_over_c6"] = np.where(np.isfinite(c8) & np.isfinite(c6), c8 / np.maximum(np.abs(c6), 1e-5), np.nan)
        x = pd.concat([x, derived.astype(np.float32)], axis=1)
    return x, list(x.columns)


def rank01(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average").astype(np.float64) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prefix", default="domain65k_d_full_feature_search")
    parser.add_argument("--fusion-prefix", default="domain65k_d_crossfit_fusion")
    parser.add_argument("--output-prefix", default="domain65k_d_locked_final")
    args = parser.parse_args()

    base_oof = pd.read_pickle(RUN_DIR / f"{args.input_prefix}_oof.pkl")
    base_metrics = pd.read_csv(RUN_DIR / f"{args.input_prefix}_metrics.csv")
    tuning = pd.read_csv(RUN_DIR / f"{args.input_prefix}_inner_tuning.csv")
    fusion_oof = pd.read_pickle(RUN_DIR / f"{args.fusion_prefix}_oof.pkl")
    fusion_weights = pd.read_csv(RUN_DIR / f"{args.fusion_prefix}_weights.csv")
    required_folds = set(range(5))
    slots = sorted(
        c.removeprefix("pred__")
        for c in base_oof.columns
        if c.startswith("pred__") and c != "pred__" and base_oof[c].notna().all()
    )
    if not slots:
        raise RuntimeError("Base OOF is incomplete; locked test remains unopened.")
    final_metrics = base_metrics.loc[base_metrics["slot"].isin(slots)]
    if set(final_metrics["fold"].astype(int)) != required_folds or set(final_metrics["slot"]) != set(slots):
        raise RuntimeError("Base metrics do not contain all five folds and all members.")
    if set(fusion_oof["cv_fold"].astype(int)) != required_folds or not fusion_oof["score__crossfit_fusion"].notna().all():
        raise RuntimeError("Cross-fitted fusion is incomplete; locked test remains unopened.")
    if set(fusion_weights["held_fold"].astype(int)) != required_folds:
        raise RuntimeError("Fusion weights are incomplete; locked test remains unopened.")

    locked = pd.read_csv(LOCKED, low_memory=False)
    cv = pd.read_csv(DOMAIN / "domain65k_cv_pool_complete_features.csv", low_memory=False)
    if set(cv["canonical_smiles"]) & set(locked["canonical_smiles"]):
        raise RuntimeError("Canonical SMILES overlap between CV and locked test.")
    blocks = ["cheap", "single", "mc3"]
    cv_x, features = train.load_matrix(blocks)
    locked_x, locked_features = build_feature_frame(locked, blocks)
    if features != locked_features:
        missing = sorted(set(features) - set(locked_features))[:10]
        extra = sorted(set(locked_features) - set(features))[:10]
        raise RuntimeError(f"Feature mismatch. missing={missing} extra={extra}")

    x_cv = cv_x[features].to_numpy(np.float32)
    y_cv = cv_x["D"].to_numpy(float)
    folds = cv_x["cv_fold"].to_numpy(int)
    x_locked = locked_x[features].to_numpy(np.float32)
    test_member_ranks: dict[str, list[np.ndarray]] = {slot: [] for slot in slots}
    test_member_raw: dict[str, list[np.ndarray]] = {slot: [] for slot in slots}
    model_rows: list[dict] = []

    for outer in range(5):
        train_idx = np.flatnonzero(folds != outer)
        order = train.feature_order(x_cv[train_idx], y_cv[train_idx])
        for slot in slots:
            record = final_metrics.loc[(final_metrics["fold"] == outer) & (final_metrics["slot"] == slot)]
            if len(record) != 1:
                raise RuntimeError(f"Expected exactly one frozen metric record for fold={outer}, slot={slot}.")
            record = record.iloc[0]
            config = json.loads(record["config_json"])
            k = config["top_k"] or len(features)
            selected = order[: min(int(k), len(features))]
            pred = train.refit_predict(
                slot, config, int(record["best_iteration"]), x_cv[train_idx][:, selected], y_cv[train_idx],
                x_locked[:, selected], train.SEED + outer * 1000 + len(slot),
            )
            test_member_raw[slot].append(pred)
            test_member_ranks[slot].append(rank01(pred))
            model_rows.append(
                {"outer_fold": outer, "slot": slot, "config_json": record["config_json"],
                 "best_iteration": int(record["best_iteration"]), "n_features": int(len(selected))}
            )
            print(f"locked predict fold={outer} slot={slot}", flush=True)

    member_scores = pd.DataFrame(index=locked.index)
    for slot in slots:
        # Different target transforms have arbitrary score scales. Rank each
        # constituent on the locked test before ensembling it with peers.
        member_scores[f"score__{slot}"] = np.mean(test_member_ranks[slot], axis=0)

    weights_by_fold: dict[int, dict[str, float]] = {}
    for _, row in fusion_weights.iterrows():
        entries = json.loads(row["weights_json"])
        weights_by_fold[int(row["held_fold"])] = {
            name.removeprefix("pred__"): float(value) for name, value in entries
        }
    frozen_weights = {
        slot: float(np.mean([weights_by_fold[fold].get(slot, 0.0) for fold in range(5)]))
        for slot in slots
    }
    total = sum(frozen_weights.values())
    if total <= 0:
        raise RuntimeError("Frozen fusion weights sum to zero.")
    frozen_weights = {slot: value / total for slot, value in frozen_weights.items()}
    fused = sum(member_scores[f"score__{slot}"].to_numpy(float) * weight for slot, weight in frozen_weights.items())

    # The following is the first and only point where locked labels are read for metrics.
    y_locked = locked["D"].to_numpy(float)
    output = locked[["domain_index", "canonical_smiles", "D"]].copy()
    for slot in slots:
        output[f"score__{slot}"] = member_scores[f"score__{slot}"].to_numpy(float)
    output["score__final_fusion"] = fused
    output.to_csv(RUN_DIR / f"{args.output_prefix}_predictions.csv", index=False)
    metrics = pd.DataFrame([
        {"model": "final_frozen_rank_fusion", **train.evaluate(y_locked, fused, train.SEED + 12000)},
        *[
            {"model": slot, **train.evaluate(y_locked, member_scores[f"score__{slot}"].to_numpy(float), train.SEED + 12001 + i)}
            for i, slot in enumerate(slots)
        ],
    ])
    metrics.to_csv(RUN_DIR / f"{args.output_prefix}_metrics.csv", index=False)
    pd.DataFrame(model_rows).to_csv(RUN_DIR / f"{args.output_prefix}_model_manifest.csv", index=False)
    manifest = {
        "purpose": "One-shot locked-test evaluation after all model decisions were frozen on CV OOF.",
        "input_prefix": args.input_prefix,
        "fusion_prefix": args.fusion_prefix,
        "n_cv": int(len(cv_x)),
        "n_locked": int(len(locked)),
        "canonical_smiles_overlap": 0,
        "blocks": blocks,
        "forbidden_inputs": "No DFT q+/q-/HOMO/LUMO/NPA/DFT xyz inputs.",
        "fusion_weight_policy": "Mean of five OOF cross-fitted weight vectors; no locked-label tuning.",
        "frozen_weights": frozen_weights,
        "locked_labels_used_only_for_final_metrics": True,
    }
    (RUN_DIR / f"{args.output_prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
