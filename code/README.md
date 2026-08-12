# D-P Compass code

This directory contains the final evaluation, figure-generation, training, refitting, prediction, and fusion scripts used for the D-P Compass workflow.

## Contents

- `analysis_outputs/`: scripts for the D model, the P model, the equivariant vector branch, frozen rank fusion, final evaluation, and manuscript figures.
- `frozen_configs/`: final feature manifests, final D-fusion weights, and vector-training provenance.

Scripts that only regenerated exploratory cohorts or manuscript figures from unreleased intermediate stores are intentionally omitted. The public package contains the final evaluation and validation entry points, alongside the retained final model-development code.

## Scope

The public final evaluation and validation plots read only the released tables under `data/`: the fixed internal Test-4000 cohort and the nested external cohorts (broad n=140; strict n=34). The D metric always uses `score__final_fusion_multiobjective`, the rank-normalized eight-member fusion selected by the manuscript's joint Spearman/NDCG@10% procedure. The older manual-fusion column in `test4000_predictions.csv` is retained only as a legacy audit field and is not used by public entry points.

The repository does not distribute data-selection scripts or the original feature-generation pipeline. The retained model-development/refitting scripts document the exact training procedure and frozen configurations, but full re-execution of the equivariant vector branch requires the three-conformer atom-graph/xTB stores generated during the study; those intermediate artifacts are not included in this release.

The public data package provides the complete development matrix, the frozen 4,000-molecule internal test set, external validation tables, final prediction tables, and feature dictionary. See the repository root `README.md` for data locations and provenance.

## Main entry points

- `run_domain65k_d_v2_tabular_candidate100_fold0.py`: tabular D-model development validation.
- `run_domain65k_d_vector_equivariant.py`: three-conformer equivariant vector D model.
- `search_domain65k_candidate100_fold0_multiobjective_fusion.py`: rank-normalized multi-objective D fusion.
- `refit_domain65k_d_candidate100_tabular.py` and `refit_domain65k_d_candidate100_xgb.py`: final tabular D refitting.
- `run_domain65k_p_candidate100_single_lgbm.py`: final single-LightGBM P model.
- `run_domain65k_candidate100_final_locked.py`: reproducible final evaluation on the released Test-4000 prediction table.
- `plot_external_rule_stratified_rank_validation.py`: external D validation for broad n=140 and its strict n=34 subset.
- `plot_p_rank_validation_panels.py`: Test-4000 and external P-rank validation panels.
- `plot_domain65k_consensus_interpretability.py`: Test-4000 ensemble-disagreement diagnostic.

## Python environment

The workflow was run with Python 3.11 and requires `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `torch`, `rdkit`, `h5py`, `matplotlib`, `networkx`, `adjustText`, and `shap` for the relevant scripts.

## Reproducing the published evaluations

After cloning the repository, run the following from the repository root:

```bash
python code/analysis_outputs/run_domain65k_candidate100_final_locked.py
python code/analysis_outputs/plot_external_rule_stratified_rank_validation.py
python code/analysis_outputs/plot_p_rank_validation_panels.py
python code/analysis_outputs/plot_domain65k_consensus_interpretability.py
```

Metrics are written under `results/`; figure scripts write vector and raster outputs under `results/figures/`. These public entry points use only released prediction tables and do not alter the frozen models or published data.

## Re-running model development

The retained training scripts are supplied for method transparency. To re-run them, place the study-generated complete feature blocks, candidate-overlap audit, vector-alignment table, and three-conformer atom-graph/xTB stores under a local directory and set `DP_COMPASS_STUDY_ARTIFACT_ROOT` to that directory. These intermediate artifacts are not part of the public release because the repository intentionally publishes the final feature matrices and final evaluation tables rather than the upstream feature-generation workflow.
