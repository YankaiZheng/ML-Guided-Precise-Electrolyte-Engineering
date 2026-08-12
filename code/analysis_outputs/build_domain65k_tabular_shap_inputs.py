#!/usr/bin/env python3
"""Build the frozen 4,000-molecule feature matrix for tabular TreeSHAP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "analysis_outputs/qme14s_training/domain65k"
RUN = DOMAIN / "model_runs"
FEATURES = DOMAIN / "features"
DERIVED = DOMAIN / "features_v2_atom3d_derived/chunks"
EVAL = RUN / "domain65k_d_candidate100_fold0_fusion_multiobjective/evaluation4000_predictions.csv"
TAB_MANIFEST = RUN / "domain65k_d_candidate100_tabular_full/manifest.json"
XGB_MANIFEST = RUN / "domain65k_d_candidate100_xgb_full/manifest.json"
OUT = ROOT / "analysis_outputs/paper_figures_ml_workflow/source_data"
OUTPUT = OUT / "feature_interpretability_input.pkl.gz"


def numeric_block(frame: pd.DataFrame, prefix: str, excluded: set[str]) -> pd.DataFrame:
    historical_metadata = {
        "v3__heavy_atoms",
        "v3__mol_wt",
        "v3__rotatable_bonds",
        "v3__ring_count",
        "cheap_feature_reused_from_16k",
        "cheap_feature_error",
        "xtb_reused_from_16k",
    }
    columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
        and column not in historical_metadata
        and not column.endswith("_selection")
        and "source_index" not in column.lower()
        and "cv_fold" not in column.lower()
    ]
    result = frame[columns].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    result.columns = [f"{prefix}{column}" for column in result.columns]
    return result


def model_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted({str(name) for member in payload["members"] for name in member["feature_columns"]})


def load_atom3d(wanted: set[int]) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for path in sorted(DERIVED.glob("chunk_*.pkl")):
        block = pd.read_pickle(path)
        keep = block["domain_index"].astype(int).isin(wanted)
        if keep.any():
            selected.append(block.loc[keep].copy())
    atom = pd.concat(selected, ignore_index=True)
    if atom["domain_index"].duplicated().any():
        raise RuntimeError("Duplicate atom3d rows in explanation cohort")
    if set(atom["domain_index"].astype(int)) != wanted:
        missing = sorted(wanted - set(atom["domain_index"].astype(int)))[:10]
        raise RuntimeError(f"Missing atom3d rows: {missing}")
    if "status" in atom and not atom["status"].eq("ok").all():
        raise RuntimeError("Non-ok atom3d row in explanation cohort")
    return atom.set_index("domain_index").drop(columns=["status"], errors="ignore")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_csv(
        EVAL,
        usecols=["domain_index", "canonical_smiles", "heavy_atoms", "mol_wt"],
    ).sort_values("domain_index").reset_index(drop=True)
    if len(evaluation) != 4000:
        raise RuntimeError("Expected 4,000 evaluation records")

    wanted = evaluation["domain_index"].astype(int).to_numpy()
    wanted_set = set(wanted.tolist())
    excluded = {"domain_index", "D", "P", "cv_fold"}

    cheap_all = pd.read_pickle(FEATURES / "domain65k_cheap_v3_v4_v5_features.pkl").set_index("domain_index")
    cheap = numeric_block(cheap_all.loc[wanted].reset_index(), "cheap__", excluded)
    del cheap_all

    single_all = pd.read_pickle(FEATURES / "domain65k_xtb_single_features.pkl").set_index("domain_index")
    single_rows = single_all.loc[wanted].reset_index()
    if not single_rows["status"].eq("ok").all():
        raise RuntimeError("Non-ok single-conformer row in explanation cohort")
    single = numeric_block(single_rows, "single__", excluded | {"xtb_reused_from_16k"})
    del single_all

    mc3_all = pd.read_pickle(FEATURES / "domain65k_xtb_mc3_features.pkl").set_index("domain_index")
    mc3_rows = mc3_all.loc[wanted].reset_index()
    complete = (
        mc3_rows["status"].eq("ok")
        & mc3_rows["xtb_mc3_n_conformer_success"].ge(3)
        & mc3_rows["n_conformer_failure"].eq(0)
    )
    if not complete.all():
        raise RuntimeError("Incomplete MC3 row in explanation cohort")
    mc3 = numeric_block(mc3_rows, "mc3__", excluded | {"xtb_reused_from_16k"})
    del mc3_all

    base = pd.concat([cheap, single, mc3], axis=1)
    heavy = evaluation["heavy_atoms"].to_numpy(np.float32)
    mw = evaluation["mol_wt"].to_numpy(np.float32)

    def value(name: str) -> np.ndarray:
        return base[f"single__{name}"].to_numpy(np.float32)

    mu = value("xtb_full_dipole_debye")
    qmu = value("xtb_qonly_dipole_debye")
    alpha = value("xtb_mol_polarizability_au")
    c6 = value("xtb_mol_c6")
    c8 = value("xtb_mol_c8")
    derived = pd.DataFrame({
        "derived__mu_per_heavy": mu / np.maximum(heavy, 1),
        "derived__mu_per_mw": mu / np.maximum(mw, 1),
        "derived__mu_minus_qmu_abs": np.abs(mu - qmu),
        "derived__qmu_over_mu": qmu / np.maximum(np.abs(mu), 1e-5),
        "derived__alpha_per_heavy": alpha / np.maximum(heavy, 1),
        "derived__alpha_per_mw": alpha / np.maximum(mw, 1),
        "derived__c6_per_heavy": c6 / np.maximum(heavy, 1),
        "derived__c8_over_c6": c8 / np.maximum(np.abs(c6), 1e-5),
    }, dtype=np.float32)

    atom = load_atom3d(wanted_set).loc[wanted].reset_index(drop=True)
    augmented = pd.concat([base, derived, atom], axis=1)
    required = sorted(set(model_features(TAB_MANIFEST)) | set(model_features(XGB_MANIFEST)))
    missing = sorted(set(required) - set(augmented.columns))
    if missing:
        raise RuntimeError(f"Missing frozen model features: {missing[:20]}")

    result = pd.concat([evaluation, augmented[required].astype(np.float32)], axis=1)
    result.to_pickle(OUTPUT, compression="gzip")
    (OUT / "feature_interpretability_input_manifest.json").write_text(
        json.dumps({
            "n_rows": len(result),
            "n_unique_domain_index": int(result["domain_index"].nunique()),
            "n_features": len(required),
            "cohort": "frozen chemistry-curated evaluation4000",
            "labels_included": False,
            "feature_order": required,
        }, indent=2),
        encoding="utf-8",
    )
    print({"rows": len(result), "features": len(required), "output": str(OUTPUT)}, flush=True)


if __name__ == "__main__":
    main()
