# D-P Compass code

This directory contains the final analysis, training, refitting, prediction, fusion, and figure-generation scripts used for the D-P Compass workflow.

## Contents

- `analysis_outputs/`: scripts for the D model, the P model, the equivariant vector branch, frozen rank fusion, evaluation, and manuscript figures.
- `frozen_configs/`: final feature manifests, final D-fusion weights, and vector-training provenance.

## Scope

The scripts use the published Electrolyte-65K feature tables and the project output layout. The repository does not distribute data-selection scripts or the original feature-generation pipeline. In particular, the vector workflow requires the three-conformer atom-graph/xTB stores generated during the study.

The public data package provides the complete development matrix, the frozen 4,000-molecule internal test set, external validation tables, final prediction tables, and feature dictionary. See the repository root `README.md` for data locations and provenance.

## Main entry points

- `run_domain65k_d_v2_tabular_candidate100_fold0.py`: tabular D-model development validation.
- `run_domain65k_d_vector_equivariant.py`: three-conformer equivariant vector D model.
- `search_domain65k_candidate100_fold0_multiobjective_fusion.py`: rank-normalized multi-objective D fusion.
- `refit_domain65k_d_candidate100_tabular.py` and `refit_domain65k_d_candidate100_xgb.py`: final tabular D refitting.
- `run_domain65k_p_candidate100_single_lgbm.py`: final single-LightGBM P model.
- `run_domain65k_candidate100_final_locked.py`: frozen internal evaluation.
- `plot_paper_figures_2_6.py`: manuscript figure generation.

## Python environment

The workflow was run with Python 3.11 and requires `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `torch`, `rdkit`, `h5py`, `matplotlib`, `networkx`, `adjustText`, and `shap` for the relevant scripts.
