#!/usr/bin/env python3
"""Deterministic Pareto-balanced rank fusion selected only on fold-0.

The selected point maximizes the weaker normalized attainment of Spearman and
NDCG@10% between their independently optimized endpoint solutions.  This is a
max-min compromise and does not require an arbitrary fixed metric mixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


REPOSITORY = Path(__file__).resolve().parents[2]
RUN = Path(__import__("os").environ.get("DP_COMPASS_STUDY_ARTIFACT_ROOT", REPOSITORY / "study_artifacts")) / "domain65k/model_runs"
TABULAR = RUN / "domain65k_d_v2_tabular_candidate100_fold0_frozen081/fold0_predictions.csv"
XGB = RUN / "domain65k_d_candidate100_xgb_fold0/fold0_predictions.csv"
VECTOR = RUN / "domain65k_d_vector_candidate100_gpu_fold0_lr5e5_e35_outer_eval_predictions.csv"
OUT_DIR = RUN / "domain65k_d_candidate100_fold0_fusion_multiobjective"
MAX_MEMBER_WEIGHT = 0.60
STEPS = (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005)


def consistent_unique(frame: pd.DataFrame) -> pd.DataFrame:
    duplicate = frame.loc[frame["domain_index"].duplicated(keep=False)]
    if not duplicate.empty:
        comparable = ["canonical_smiles", "D"] + [c for c in frame if c.startswith("pred__")]
        inconsistent = duplicate.groupby("domain_index")[comparable].nunique(dropna=False).max(axis=1) > 1
        if inconsistent.any():
            raise RuntimeError("Inconsistent duplicate predictions")
    return frame.drop_duplicates("domain_index", keep="first").reset_index(drop=True)


def rank01(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average").astype(np.float64) / len(values)


def ndcg10(y: np.ndarray, score: np.ndarray) -> float:
    k = int(np.ceil(len(y) * 0.10))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    pred_order = np.argsort(score)[::-1][:k]
    ideal_order = np.argsort(y)[::-1][:k]
    return float(np.sum(y[pred_order] * discount) / np.sum(y[ideal_order] * discount))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab = consistent_unique(pd.read_csv(TABULAR))
    xgb = consistent_unique(pd.read_csv(XGB))
    vec = consistent_unique(pd.read_csv(VECTOR))
    frame = tab.merge(xgb, on=["domain_index", "canonical_smiles", "D", "cv_fold"], validate="one_to_one")
    frame = frame.merge(
        vec[["domain_index", "D", "pred__vector_mu"]], on="domain_index", validate="one_to_one",
        suffixes=("_base", "_vector"),
    ).sort_values("domain_index").reset_index(drop=True)
    if not np.allclose(frame["D_base"], frame["D_vector"], atol=1e-8, rtol=0):
        raise RuntimeError("Target mismatch")
    y = frame["D_base"].to_numpy(float)
    names = sorted(c for c in frame if c.startswith("pred__") and c != "pred__vector_mu") + ["pred__vector_mu"]
    x = np.column_stack([rank01(frame[name].to_numpy(float)) for name in names])
    metric_cache: dict[tuple[float, ...], tuple[float, float]] = {}

    def metrics(weights: np.ndarray) -> tuple[float, float]:
        key = tuple(np.round(weights, 7))
        if key not in metric_cache:
            score = x @ weights
            metric_cache[key] = (float(spearmanr(y, score).statistic), ndcg10(y, score))
        return metric_cache[key]

    starts = [np.full(len(names), 1.0 / len(names))]
    starts.extend(np.eye(len(names), dtype=float))
    rng = np.random.default_rng(20260716)
    starts.extend(rng.dirichlet(np.full(len(names), 1.5), size=24))

    def coordinate_optimize(initial: np.ndarray, utility) -> tuple[np.ndarray, tuple[float, float]]:
        weights = initial.copy()
        if weights.max() > MAX_MEMBER_WEIGHT:
            excess = weights.max() - MAX_MEMBER_WEIGHT
            idx = int(np.argmax(weights))
            weights[idx] = MAX_MEMBER_WEIGHT
            recipients = np.arange(len(weights)) != idx
            recipient_sum = weights[recipients].sum()
            if recipient_sum > 0:
                weights[recipients] += excess * weights[recipients] / recipient_sum
            else:
                weights[recipients] = excess / int(recipients.sum())
        value = metrics(weights)
        best_u = utility(value, weights)
        for delta in STEPS:
            improved = True
            while improved:
                improved = False
                best_candidate = None
                for source in range(len(weights)):
                    if weights[source] + 1e-12 < delta:
                        continue
                    for target in range(len(weights)):
                        if source == target or weights[target] + delta > MAX_MEMBER_WEIGHT + 1e-12:
                            continue
                        candidate = weights.copy()
                        candidate[source] -= delta
                        candidate[target] += delta
                        candidate_value = metrics(candidate)
                        candidate_u = utility(candidate_value, candidate)
                        if candidate_u > best_u:
                            best_candidate = (candidate_u, candidate, candidate_value)
                            best_u = candidate_u
                if best_candidate is not None:
                    _, weights, value = best_candidate
                    improved = True
        return weights, value

    spearman_utility = lambda value, weights: (value[0], value[1], -float(np.sum(weights * weights)))
    ndcg_utility = lambda value, weights: (value[1], value[0], -float(np.sum(weights * weights)))
    endpoint_s = max((coordinate_optimize(start, spearman_utility) for start in starts), key=lambda item: spearman_utility(item[1], item[0]))
    endpoint_n = max((coordinate_optimize(start, ndcg_utility) for start in starts), key=lambda item: ndcg_utility(item[1], item[0]))
    s_max, n_at_smax = endpoint_s[1]
    s_at_nmax, n_max = endpoint_n[1]
    s_span = max(s_max - s_at_nmax, 1e-9)
    n_span = max(n_max - n_at_smax, 1e-9)

    def attainment(value: tuple[float, float]) -> tuple[float, float]:
        return ((value[0] - s_at_nmax) / s_span, (value[1] - n_at_smax) / n_span)

    candidates: list[dict] = []
    frontier_starts = [endpoint_s[0], endpoint_n[0], old]
    for alpha in np.linspace(0.0, 1.0, 21):
        initial = endpoint_n[0] * (1.0 - alpha) + endpoint_s[0] * alpha
        local_starts = [initial, *frontier_starts]

        def scalar_utility(value, weights, alpha=float(alpha)):
            us, un = attainment(value)
            return (alpha * us + (1.0 - alpha) * un, min(us, un), -float(np.sum(weights * weights)))

        result = max((coordinate_optimize(start, scalar_utility) for start in local_starts),
                     key=lambda item: scalar_utility(item[1], item[0]))
        us, un = attainment(result[1])
        candidates.append({
            "alpha": float(alpha), "weights": result[0], "spearman": result[1][0],
            "ndcg_at_10pct": result[1][1], "spearman_attainment": us,
            "ndcg_attainment": un, "min_attainment": min(us, un),
            "mean_attainment": 0.5 * (us + un), "weight_l2": float(np.sum(result[0] ** 2)),
        })

    # Remove dominated points before applying the max-min compromise rule.
    for row in candidates:
        row["pareto"] = not any(
            other["spearman"] >= row["spearman"] - 1e-12
            and other["ndcg_at_10pct"] >= row["ndcg_at_10pct"] - 1e-12
            and (other["spearman"] > row["spearman"] + 1e-12 or other["ndcg_at_10pct"] > row["ndcg_at_10pct"] + 1e-12)
            for other in candidates
        )
    pareto = [row for row in candidates if row["pareto"]]
    selected = max(pareto, key=lambda row: (row["min_attainment"], row["mean_attainment"], -row["weight_l2"]))

    # Optional interpretability pruning, accepted only when both metrics move by <= 1e-4.
    pruned = selected["weights"].copy()
    pruned[pruned < 0.005] = 0.0
    pruned /= pruned.sum()
    pruned_metrics = metrics(pruned)
    pruning_accepted = (
        selected["spearman"] - pruned_metrics[0] <= 1e-4
        and selected["ndcg_at_10pct"] - pruned_metrics[1] <= 1e-4
    )
    final_weights = pruned if pruning_accepted else selected["weights"]
    final_metrics = metrics(final_weights)
    final_attainment = attainment(final_metrics)

    table = pd.DataFrame([{k: v for k, v in row.items() if k != "weights"} | {
        **{f"weight__{name.removeprefix('pred__')}": float(weight) for name, weight in zip(names, row["weights"])},
    } for row in candidates]).sort_values(["pareto", "min_attainment"], ascending=False)
    table.to_csv(OUT_DIR / "fold0_pareto_search.csv", index=False)
    score = x @ final_weights
    prediction = frame[["domain_index", "canonical_smiles", "D_base"]].rename(columns={"D_base": "D"})
    prediction["score__multiobjective_fusion"] = score
    prediction.to_csv(OUT_DIR / "fold0_predictions.csv", index=False)
    payload = {
        "status": "fold0_multiobjective_fusion_frozen",
        "selection_rule": {
            "objectives": ["Spearman", "NDCG@10%"],
            "weight_constraints": f"non-negative simplex; each member <= {MAX_MEMBER_WEIGHT:.2f}",
            "endpoint_normalization": "Each objective is normalized between its values at the two independently optimized endpoints.",
            "compromise": "Select Pareto point maximizing the minimum normalized attainment; tie-break by mean attainment, then lower weight L2.",
            "pruning": "Weights below 0.005 are removed only if both metric decreases are <= 1e-4.",
        },
        "n_shared_graph_valid": len(frame),
        "feature_names": names,
        "spearman_endpoint": {"spearman": s_max, "ndcg_at_10pct": n_at_smax},
        "ndcg_endpoint": {"spearman": s_at_nmax, "ndcg_at_10pct": n_max},
        "selected_alpha_is_diagnostic_only": selected["alpha"],
        "selected_before_pruning": {k: float(v) for k, v in selected.items() if k not in {"weights", "pareto"}},
        "pruning_accepted": pruning_accepted,
        "selected": {
            "spearman": final_metrics[0], "ndcg_at_10pct": final_metrics[1],
            "spearman_attainment": final_attainment[0], "ndcg_attainment": final_attainment[1],
        },
        "weights_by_member": {name: float(weight) for name, weight in zip(names, final_weights)},
        "search_evaluations": len(metric_cache),
        "locked_or_evaluation_labels_read": False,
    }
    (OUT_DIR / "multiobjective_weights.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
