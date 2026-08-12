from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import time

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, rankdata, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
DOMAIN_DIR = ROOT / "analysis_outputs" / "qme14s_training" / "domain65k"
FEATURE_DIR = DOMAIN_DIR / "features"
RUN_DIR = DOMAIN_DIR / "model_runs"
CV_POOL = DOMAIN_DIR / "domain65k_cv_pool_approx60800.csv"
CHEAP = FEATURE_DIR / "domain65k_cheap_v3_v4_v5_features.pkl"
SINGLE = FEATURE_DIR / "domain65k_xtb_single_features.pkl"
MC3 = FEATURE_DIR / "domain65k_xtb_mc3_features.pkl"
COHORT_EXCLUSION = ROOT / "analysis_outputs/candidate_curation_v2/domain65k_candidate_exact_exclusion.csv"

SEED = 20260711
PAIRWISE_SAMPLE_N = 300_000
SENTINEL = -9999.0
N_THREADS = int(os.environ.get("D_MODEL_THREADS", "8"))


def finite(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def rank_pct(y: np.ndarray) -> np.ndarray:
    return pd.Series(y).rank(method="average", pct=True).to_numpy(dtype=np.float32)


def normal_score(y: np.ndarray) -> np.ndarray:
    ranks = rank_pct(y)
    return norm.ppf(np.clip(ranks, 1e-4, 1 - 1e-4)).astype(np.float32)


def transform_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return y.astype(np.float32)
    if mode == "rank":
        return rank_pct(y)
    if mode == "normal":
        return normal_score(y)
    raise ValueError(mode)


def ltr_relevance(y: np.ndarray, n_levels: int = 31) -> np.ndarray:
    """Convert continuous D into ordered relevance levels for LambdaRank."""
    return np.minimum((rank_pct(y) * n_levels).astype(np.int32), n_levels - 1)


def ltr_query_permutation(n_rows: int, seed: int, max_query_size: int = 8000) -> tuple[np.ndarray, list[int]]:
    """Create deterministic random query blocks below LightGBM's 10k hard limit."""
    order = np.random.default_rng(seed).permutation(n_rows)
    groups = [min(max_query_size, n_rows - start) for start in range(0, n_rows, max_query_size)]
    return order, groups


def ndcg(y: np.ndarray, score: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(math.ceil(len(y) * frac)))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    top = np.argsort(score)[::-1][:k]
    ideal = np.argsort(y)[::-1][:k]
    denom = float((y[ideal] * discount).sum())
    return finite(float((y[top] * discount).sum()) / denom) if denom else float("nan")


def top_overlap(y: np.ndarray, score: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(math.ceil(len(y) * frac)))
    true_top = set(np.argsort(y)[::-1][:k])
    pred_top = set(np.argsort(score)[::-1][:k])
    return float(len(true_top & pred_top) / k)


def pairwise_accuracy(y: np.ndarray, score: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(y), PAIRWISE_SAMPLE_N)
    right = rng.integers(0, len(y), PAIRWISE_SAMPLE_N)
    mask = left != right
    yd = y[left[mask]] - y[right[mask]]
    sd = score[left[mask]] - score[right[mask]]
    mask = yd != 0
    return float(np.mean(np.sign(yd[mask]) == np.sign(sd[mask]))) if np.any(mask) else float("nan")


def evaluate(y: np.ndarray, score: np.ndarray, seed: int) -> dict[str, float]:
    overlap05 = top_overlap(y, score, 0.05)
    overlap10 = top_overlap(y, score, 0.10)
    overlap20 = top_overlap(y, score, 0.20)
    calibrated = np.poly1d(np.polyfit(score, y, 1))(score) if len(np.unique(score)) > 1 else score
    return {
        "spearman": finite(spearmanr(y, score).statistic),
        "kendall_tau": finite(kendalltau(y, score).statistic),
        "pairwise_accuracy": pairwise_accuracy(y, score, seed),
        "ndcg_at_05pct": ndcg(y, score, 0.05),
        "ndcg_at_10pct": ndcg(y, score, 0.10),
        "ndcg_at_20pct": ndcg(y, score, 0.20),
        "ef_at_05pct": float(overlap05 / 0.05),
        "ef_at_10pct": float(overlap10 / 0.10),
        "ef_at_20pct": float(overlap20 / 0.20),
        "top_overlap_at_05pct": overlap05,
        "top_overlap_at_10pct": overlap10,
        "top_overlap_at_20pct": overlap20,
        "mae_calibrated": finite(mean_absolute_error(y, calibrated)),
    }


def numeric_block(df: pd.DataFrame, prefix: str, excluded: set[str], fill_failure: bool) -> pd.DataFrame:
    historical_metadata = {
        "v3__heavy_atoms",
        "v3__mol_wt",
        "v3__rotatable_bonds",
        "v3__ring_count",
        "cheap_feature_reused_from_16k",
        "cheap_feature_error",
        "xtb_reused_from_16k",
    }
    cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in excluded
        and c not in historical_metadata
        and not c.endswith("_selection")
        and "source_index" not in c.lower()
        and "cv_fold" not in c.lower()
    ]
    out = df[cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    if fill_failure:
        out = out.fillna(SENTINEL)
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def candidate_exclusion_indices() -> set[int]:
    """Exact candidate structures are never used as development or locked rows."""
    if not COHORT_EXCLUSION.exists():
        raise FileNotFoundError(f"Missing candidate exclusion audit: {COHORT_EXCLUSION}")
    table = pd.read_csv(COHORT_EXCLUSION, usecols=["domain_index"])
    if table["domain_index"].duplicated().any():
        raise RuntimeError("Candidate exclusion audit contains duplicate domain_index values")
    return set(table["domain_index"].astype(int))


def load_matrix(blocks: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cv = pd.read_csv(CV_POOL, low_memory=False)
    domain = pd.read_pickle(CHEAP)
    domain_index = domain["domain_index"].to_numpy(int)
    index_by_smiles = pd.Series(domain_index, index=domain["canonical_smiles"].astype(str)).groupby(level=0).first()
    cv = cv.copy()
    cv["domain_index"] = cv["canonical_smiles"].astype(str).map(index_by_smiles)
    if cv["domain_index"].isna().any():
        raise ValueError("CV rows missing from feature-domain index.")
    candidate_exclusions = candidate_exclusion_indices()
    candidate_keep = ~cv["domain_index"].astype(int).isin(candidate_exclusions)
    if not candidate_keep.all():
        print(
            f"candidate-overlap filter: kept={int(candidate_keep.sum())} "
            f"dropped={int((~candidate_keep).sum())}",
            flush=True,
        )
        cv = cv.loc[candidate_keep].reset_index(drop=True)
    wanted = cv["domain_index"].astype(int).to_numpy()

    # The analysis protocol uses complete molecules only. A molecule that failed
    # either requested xTB calculation is excluded instead of being represented
    # by an imputed value or a sentinel.
    aligned_single = None
    aligned_mc3 = None
    keep = np.ones(len(cv), dtype=bool)
    if "single" in blocks:
        aligned_single = pd.read_pickle(SINGLE).set_index("domain_index").loc[wanted].reset_index()
        keep &= aligned_single["status"].eq("ok").to_numpy()
    if "mc3" in blocks:
        aligned_mc3 = pd.read_pickle(MC3).set_index("domain_index").loc[wanted].reset_index()
        keep &= (
            aligned_mc3["status"].eq("ok")
            & aligned_mc3["xtb_mc3_n_conformer_success"].ge(3)
        ).to_numpy()
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"complete-case filter: kept={int(keep.sum())} dropped={n_dropped}", flush=True)
    cv = cv.loc[keep].reset_index(drop=True)
    wanted = wanted[keep]
    if aligned_single is not None:
        aligned_single = aligned_single.loc[keep].reset_index(drop=True)
    if aligned_mc3 is not None:
        aligned_mc3 = aligned_mc3.loc[keep].reset_index(drop=True)

    meta = cv[["canonical_smiles", "D", "P", "cv_fold", "domain_index"]].copy()
    feature_parts = []
    excluded = {"domain_index", "D", "P", "cv_fold"}
    if "cheap" in blocks:
        cheap = domain.set_index("domain_index").loc[wanted].reset_index()
        feature_parts.append(numeric_block(cheap, "cheap__", excluded, fill_failure=False))
    if "single" in blocks:
        part = numeric_block(aligned_single, "single__", excluded | {"xtb_reused_from_16k"}, fill_failure=False)
        feature_parts.append(part)
    if "mc3" in blocks:
        part = numeric_block(aligned_mc3, "mc3__", excluded | {"xtb_reused_from_16k"}, fill_failure=False)
        feature_parts.append(part)

    x = pd.concat([part.reset_index(drop=True) for part in feature_parts], axis=1)
    heavy = cv["heavy_atoms"].to_numpy(np.float32)
    mw = cv["mol_wt"].to_numpy(np.float32)
    if "single" in blocks:
        def col(name: str) -> np.ndarray:
            key = f"single__{name}"
            return x[key].to_numpy(np.float32) if key in x else np.full(len(x), np.nan, dtype=np.float32)

        mu = col("xtb_full_dipole_debye")
        qmu = col("xtb_qonly_dipole_debye")
        alpha = col("xtb_mol_polarizability_au")
        c6 = col("xtb_mol_c6")
        c8 = col("xtb_mol_c8")
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
    frame = pd.concat([meta.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    features = list(x.columns)
    if len(features) != len(set(features)):
        duplicates = pd.Series(features).value_counts()
        raise ValueError(f"Duplicate features: {duplicates[duplicates > 1].index.tolist()[:20]}")
    return frame, features


def feature_order(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    target = rank_pct(y).astype(np.float64)
    target -= target.mean()
    data = x.astype(np.float64, copy=False)
    scores = np.zeros(data.shape[1], dtype=np.float64)
    for start in range(0, data.shape[1], 256):
        block = data[:, start : start + 256]
        means = np.nanmean(block, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        filled = np.where(np.isfinite(block), block, means)
        filled -= filled.mean(axis=0)
        denom = np.sqrt(np.sum(filled * filled, axis=0) * np.sum(target * target))
        corr = np.divide(target @ filled, denom, out=np.zeros_like(denom), where=denom > 0)
        scores[start : start + len(corr)] = np.abs(corr)
    return np.argsort(scores)[::-1]


def candidate_configs(slot: str) -> list[dict]:
    if slot.startswith("lgbm"):
        return [
            {"top_k": None, "leaves": 95, "min_child": 20, "lr": 0.025, "cols": 0.82},
            {"top_k": 2048, "leaves": 127, "min_child": 16, "lr": 0.022, "cols": 0.76},
            {"top_k": 1024, "leaves": 159, "min_child": 14, "lr": 0.019, "cols": 0.72},
            {"top_k": 512, "leaves": 63, "min_child": 24, "lr": 0.030, "cols": 0.88},
        ]
    if slot.startswith("xgb"):
        return [
            {"top_k": 2048, "depth": 7, "min_child": 8, "lr": 0.030, "cols": 0.78},
            {"top_k": 1024, "depth": 8, "min_child": 10, "lr": 0.024, "cols": 0.72},
            {"top_k": 512, "depth": 6, "min_child": 6, "lr": 0.035, "cols": 0.85},
        ]
    if slot.startswith("cat"):
        return [
            {"top_k": 2048, "depth": 8, "lr": 0.035, "l2": 5.0},
            {"top_k": 1024, "depth": 9, "lr": 0.028, "l2": 7.0},
            {"top_k": 512, "depth": 7, "lr": 0.045, "l2": 4.0},
        ]
    return [
        {"top_k": 2048, "leaf": 2, "max_features": 0.8},
        {"top_k": 1024, "leaf": 2, "max_features": 1.0},
        {"top_k": 512, "leaf": 1, "max_features": 1.0},
    ]


def fit_predict(slot: str, config: dict, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    target_mode = slot.split("_")[-1]
    family = slot.split("_")[0]
    if family == "lgbm" and target_mode == "ltr":
        target = ltr_relevance(y_train)
        val_target = ltr_relevance(y_val)
        train_order, train_groups = ltr_query_permutation(len(y_train), seed + 11)
        val_order, val_groups = ltr_query_permutation(len(y_val), seed + 29)
        model = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", label_gain=list(range(32)),
            lambdarank_truncation_level=800,
            n_estimators=1800, learning_rate=config["lr"], num_leaves=config["leaves"],
            min_child_samples=config["min_child"], subsample=0.92, colsample_bytree=config["cols"],
            reg_alpha=0.04, reg_lambda=0.16, max_bin=255, force_col_wise=True,
            random_state=seed, n_jobs=N_THREADS, verbose=-1,
        )
        eval_k = max(10, int(math.ceil(len(y_val) * 0.10)))
        model.fit(
            x_train[train_order], target[train_order], group=train_groups,
            eval_set=[(x_val[val_order], val_target[val_order])], eval_group=[val_groups], eval_at=[min(eval_k, 800)],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        return np.asarray(model.predict(x_val), float), int(model.best_iteration_ or 1800)
    target = transform_target(y_train, target_mode)
    if family == "lgbm":
        model = lgb.LGBMRegressor(
            objective="regression", n_estimators=1800, learning_rate=config["lr"], num_leaves=config["leaves"],
            min_child_samples=config["min_child"], subsample=0.92, colsample_bytree=config["cols"],
            reg_alpha=0.04, reg_lambda=0.16, max_bin=255, force_col_wise=True,
            random_state=seed, n_jobs=N_THREADS, verbose=-1,
        )
        model.fit(x_train, target, eval_set=[(x_val, transform_target(y_val, target_mode))], callbacks=[lgb.early_stopping(80, verbose=False)])
        return np.asarray(model.predict(x_val), float), int(model.best_iteration_ or 1800)
    if family == "xgb":
        model = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=1600, learning_rate=config["lr"], max_depth=config["depth"],
            min_child_weight=config["min_child"], subsample=0.90, colsample_bytree=config["cols"],
            reg_alpha=0.04, reg_lambda=0.70, tree_method="hist", random_state=seed, n_jobs=N_THREADS,
            early_stopping_rounds=80, verbosity=0,
        )
        model.fit(x_train, target, eval_set=[(x_val, transform_target(y_val, target_mode))], verbose=False)
        return np.asarray(model.predict(x_val), float), int(getattr(model, "best_iteration", 1599) + 1)
    if family == "cat":
        model = cb.CatBoostRegressor(
            loss_function="RMSE", iterations=1400, learning_rate=config["lr"], depth=config["depth"],
            l2_leaf_reg=config["l2"], random_seed=seed, thread_count=N_THREADS, verbose=False, allow_writing_files=False,
            od_type="Iter", od_wait=80,
        )
        model.fit(x_train, target, eval_set=(x_val, transform_target(y_val, target_mode)), verbose=False)
        return np.asarray(model.predict(x_val), float), int(model.get_best_iteration() + 1)
    finite_cols = np.isfinite(x_train).all(axis=0) & np.isfinite(x_val).all(axis=0)
    if not np.any(finite_cols):
        raise ValueError("ExtraTrees has no fully observed feature columns.")
    x_train = x_train[:, finite_cols]
    x_val = x_val[:, finite_cols]
    model = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=config["leaf"], max_features=config["max_features"],
        random_state=seed, n_jobs=N_THREADS,
    )
    model.fit(x_train, target)
    return np.asarray(model.predict(x_val), float), 500


def refit_predict(slot: str, config: dict, n_iter: int, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, seed: int) -> np.ndarray:
    mode = slot.split("_")[-1]
    family = slot.split("_")[0]
    if family == "lgbm" and mode == "ltr":
        target = ltr_relevance(y_train)
        train_order, train_groups = ltr_query_permutation(len(y_train), seed + 11)
        model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", label_gain=list(range(32)), lambdarank_truncation_level=800, n_estimators=max(50, n_iter), learning_rate=config["lr"], num_leaves=config["leaves"], min_child_samples=config["min_child"], subsample=0.92, colsample_bytree=config["cols"], reg_alpha=0.04, reg_lambda=0.16, max_bin=255, force_col_wise=True, random_state=seed, n_jobs=N_THREADS, verbose=-1)
        model.fit(x_train[train_order], target[train_order], group=train_groups)
        return np.asarray(model.predict(x_val), float)
    target = transform_target(y_train, mode)
    if family == "lgbm":
        model = lgb.LGBMRegressor(objective="regression", n_estimators=max(50, n_iter), learning_rate=config["lr"], num_leaves=config["leaves"], min_child_samples=config["min_child"], subsample=0.92, colsample_bytree=config["cols"], reg_alpha=0.04, reg_lambda=0.16, max_bin=255, force_col_wise=True, random_state=seed, n_jobs=N_THREADS, verbose=-1)
    elif family == "xgb":
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=max(50, n_iter), learning_rate=config["lr"], max_depth=config["depth"], min_child_weight=config["min_child"], subsample=0.90, colsample_bytree=config["cols"], reg_alpha=0.04, reg_lambda=0.70, tree_method="hist", random_state=seed, n_jobs=N_THREADS, verbosity=0)
    elif family == "cat":
        model = cb.CatBoostRegressor(loss_function="RMSE", iterations=max(50, n_iter), learning_rate=config["lr"], depth=config["depth"], l2_leaf_reg=config["l2"], random_seed=seed, thread_count=N_THREADS, verbose=False, allow_writing_files=False)
    else:
        finite_cols = np.isfinite(x_train).all(axis=0) & np.isfinite(x_val).all(axis=0)
        if not np.any(finite_cols):
            raise ValueError("ExtraTrees has no fully observed feature columns.")
        x_train = x_train[:, finite_cols]
        x_val = x_val[:, finite_cols]
        model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=config["leaf"], max_features=config["max_features"], random_state=seed, n_jobs=N_THREADS)
    model.fit(x_train, target)
    return np.asarray(model.predict(x_val), float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["D", "P"], default="D")
    parser.add_argument("--blocks", default="cheap,single,mc3")
    parser.add_argument("--slots", default="lgbm_raw,lgbm_rank,lgbm_normal,lgbm_ltr,xgb_raw,xgb_rank,cat_raw,cat_rank,extra_raw,extra_rank")
    parser.add_argument("--output-prefix", default="domain65k_d_full_feature_search")
    parser.add_argument("--outer-folds", default="", help="Comma-separated outer folds to run; default runs all five.")
    parser.add_argument("--pilot-per-fold", type=int, default=0, help="Stratified target sample size per existing fold; 0 uses all rows.")
    parser.add_argument("--config-index", type=int, default=-1, help="Use one fixed candidate configuration by index; -1 searches all configurations.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    blocks = [x.strip() for x in args.blocks.split(",") if x.strip()]
    slots = [x.strip() for x in args.slots.split(",") if x.strip()]
    target_name = args.target
    frame, features = load_matrix(blocks)
    if args.pilot_per_fold:
        sampled = []
        for fold in sorted(frame["cv_fold"].unique()):
            part = frame.loc[frame["cv_fold"] == fold]
            n_take = min(int(args.pilot_per_fold), len(part))
            bins = pd.qcut(part[target_name].rank(method="first"), q=10, labels=False, duplicates="drop")
            rng = np.random.default_rng(SEED + int(fold))
            chosen = []
            for _, group in part.groupby(bins, observed=True):
                quota = max(1, n_take // max(1, int(bins.nunique())))
                chosen.extend(rng.choice(group.index.to_numpy(), size=min(quota, len(group)), replace=False).tolist())
            if len(chosen) < n_take:
                remainder = np.setdiff1d(part.index.to_numpy(), np.asarray(chosen, dtype=int), assume_unique=False)
                chosen.extend(rng.choice(remainder, size=n_take - len(chosen), replace=False).tolist())
            sampled.extend(chosen[:n_take])
        frame = frame.loc[sorted(sampled)].reset_index(drop=True)
        print(f"pilot sample: rows={len(frame)} per_fold={args.pilot_per_fold} target={target_name}", flush=True)
    x_all = frame[features].to_numpy(dtype=np.float32)
    y_all = frame[target_name].to_numpy(float)
    folds = frame["cv_fold"].to_numpy(int)
    oof_path = RUN_DIR / f"{args.output_prefix}_oof.pkl"
    metrics_path = RUN_DIR / f"{args.output_prefix}_metrics.csv"
    tuning_path = RUN_DIR / f"{args.output_prefix}_inner_tuning.csv"
    oof = frame[["domain_index", "canonical_smiles", target_name, "cv_fold"]].copy()
    metrics_rows: list[dict] = []
    tuning_rows: list[dict] = []
    if not args.no_resume and oof_path.exists() and metrics_path.exists():
        previous_oof = pd.read_pickle(oof_path)
        if previous_oof["domain_index"].tolist() != oof["domain_index"].tolist():
            raise ValueError("Existing OOF file is not aligned with the current complete-case dataset.")
        for col in previous_oof.columns:
            if col.startswith("pred__"):
                oof[col] = previous_oof[col].to_numpy()
        metrics_rows = pd.read_csv(metrics_path).to_dict("records")
        if tuning_path.exists():
            tuning_rows = pd.read_csv(tuning_path).to_dict("records")
        print(f"resume: loaded {len(metrics_rows)} completed fold/member results", flush=True)
    completed = {(int(row["fold"]), str(row["slot"])) for row in metrics_rows}
    started = time.time()

    available_folds = sorted(np.unique(folds))
    target_folds = available_folds if not args.outer_folds else [int(x) for x in args.outer_folds.split(",")]
    if any(fold not in available_folds for fold in target_folds):
        raise ValueError(f"Requested outer folds {target_folds} are not present in {available_folds}.")
    for outer in target_folds:
        outer_train_idx = np.flatnonzero(folds != outer)
        outer_val_idx = np.flatnonzero(folds == outer)
        inner_fold = (int(outer) + 1) % 5
        inner_train_idx = np.flatnonzero((folds != outer) & (folds != inner_fold))
        inner_val_idx = np.flatnonzero(folds == inner_fold)
        inner_order = feature_order(x_all[inner_train_idx], y_all[inner_train_idx])
        outer_order = feature_order(x_all[outer_train_idx], y_all[outer_train_idx])
        for slot in slots:
            if (int(outer), slot) in completed:
                print(f"resume skip fold={outer} slot={slot}", flush=True)
                continue
            best = None
            configs = candidate_configs(slot)
            if args.config_index >= 0:
                if args.config_index >= len(configs):
                    raise ValueError(f"config-index {args.config_index} unavailable for {slot}.")
                configs = [configs[args.config_index]]
            for config_id, config in enumerate(configs):
                k = config["top_k"] or len(features)
                selected = inner_order[: min(k, len(features))]
                pred, n_iter = fit_predict(slot, config, x_all[inner_train_idx][:, selected], y_all[inner_train_idx], x_all[inner_val_idx][:, selected], y_all[inner_val_idx], SEED + outer * 1000 + config_id * 17 + len(slot))
                result = evaluate(y_all[inner_val_idx], pred, SEED + outer * 101 + config_id)
                objective = result["spearman"] + 0.02 * result["ndcg_at_10pct"] + 0.001 * min(result["ef_at_10pct"], 5.0)
                record = {"outer_fold": int(outer), "inner_fold": int(inner_fold), "slot": slot, "config_id": int(config_id), "config_json": json.dumps(config), "n_features": int(len(selected)), "best_iteration": int(n_iter), "objective": float(objective), **result}
                tuning_rows.append(record)
                if best is None or objective > best[0]:
                    best = (objective, config, n_iter)
            assert best is not None
            _, config, n_iter = best
            k = config["top_k"] or len(features)
            selected = outer_order[: min(k, len(features))]
            pred = refit_predict(slot, config, n_iter, x_all[outer_train_idx][:, selected], y_all[outer_train_idx], x_all[outer_val_idx][:, selected], SEED + outer * 1000 + len(slot))
            pred_col = f"pred__{slot}"
            if pred_col not in oof:
                oof[pred_col] = np.nan
            oof.loc[outer_val_idx, pred_col] = pred
            result = evaluate(y_all[outer_val_idx], pred, SEED + outer * 101 + len(slot))
            metrics_rows.append({"fold": int(outer), "slot": slot, "n_train": int(len(outer_train_idx)), "n_val": int(len(outer_val_idx)), "n_features": int(len(selected)), "config_json": json.dumps(config), "best_iteration": int(n_iter), **result})
            pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False)
            pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
            oof.to_pickle(oof_path)
            print(f"fold={outer} slot={slot} sp={result['spearman']:.5f} ndcg10={result['ndcg_at_10pct']:.5f} ef10={result['ef_at_10pct']:.3f}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    summary = metrics.groupby("slot").agg(
        folds=("fold", "nunique"),
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
        kendall_mean=("kendall_tau", "mean"), pairwise_mean=("pairwise_accuracy", "mean"),
        ndcg05_mean=("ndcg_at_05pct", "mean"), ndcg10_mean=("ndcg_at_10pct", "mean"), ndcg20_mean=("ndcg_at_20pct", "mean"),
        ef05_mean=("ef_at_05pct", "mean"), ef10_mean=("ef_at_10pct", "mean"), ef20_mean=("ef_at_20pct", "mean"),
        overlap05_mean=("top_overlap_at_05pct", "mean"), overlap10_mean=("top_overlap_at_10pct", "mean"), overlap20_mean=("top_overlap_at_20pct", "mean"),
        mae_mean=("mae_calibrated", "mean"),
    ).reset_index().sort_values("spearman_mean", ascending=False)
    pooled_rows = []
    for slot in slots:
        pred_col = f"pred__{slot}"
        if pred_col not in oof or not oof[pred_col].notna().all():
            continue
        pooled_score = np.zeros(len(oof), dtype=np.float64)
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            pooled_score[mask] = rank_pct(oof.loc[mask, pred_col].to_numpy(float))
        pooled = evaluate(y_all, pooled_score, SEED + 9000 + len(slot))
        pooled_rows.append({"slot": slot, **{f"pooled_ranknorm_{k}": v for k, v in pooled.items()}})
    if pooled_rows:
        summary = summary.merge(pd.DataFrame(pooled_rows), on="slot", how="left")
        summary = summary.sort_values("pooled_ranknorm_spearman", ascending=False)
    summary.to_csv(RUN_DIR / f"{args.output_prefix}_summary.csv", index=False)
    manifest = {"created_at": pd.Timestamp.now().isoformat(), "purpose": f"Strict nested domain-only {target_name} ranking on the 60.8k electrolyte CV pool.", "target": target_name, "blocks": blocks, "slots": slots, "outer_folds_run": target_folds, "n_rows": int(len(frame)), "n_features": int(len(features)), "cv_pool": str(CV_POOL), "complete_case_policy": "Exclude molecules unless single xTB succeeds and all three mc3 conformers succeed; no failure sentinel or mean/median imputation.", "locked_test_status": "not loaded or evaluated", "feature_selection": f"Training-only absolute correlation against rank({target_name}); all/2048/1024/512 candidates.", "parameter_search": "Deterministic inner-fold candidate search; Optuna installation was unavailable due approval-service failure.", "metrics": "Spearman primary; NDCG and EF at 5/10/20 percent co-reported.", "elapsed_sec": float(time.time() - started), "python": platform.python_version(), "forbidden_inputs": "No pretraining/teacher and no DFT q+/q-/HOMO/LUMO/NPA/DFT xyz inputs."}
    (RUN_DIR / f"{args.output_prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
