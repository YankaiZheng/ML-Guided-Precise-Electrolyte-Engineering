# D-P Compass code

This directory separates the runnable public evaluation/plotting workflow from
method-reference scripts that document the final fitting protocol.

## Contents

- `analysis_outputs/`: public evaluation/plotting entry points plus retained method-reference training, refitting, and vector scripts.
- `frozen_configs/`: final feature manifests, final D-fusion weights, and vector-training provenance.

Upstream data-selection and feature-generation scripts are intentionally
omitted. The public package contains final evaluation and validation entry
points, compact figure-source tables, and retained final model-development
scripts as method references.

## Scope

The public final evaluation and plotting scripts read only released tables under
`data/`: the fixed internal Test-4000 cohort, the nested external cohorts,
the 78-candidate screen, and the compact figure-source tables. The D metric
always uses `score__final_fusion_multiobjective`, the rank-normalized
eight-member fusion selected by the manuscript's joint Spearman/NDCG@10%
procedure. The older manual-fusion column in `test4000_predictions.csv` is a
legacy audit field and is not used by public entry points.

The repository does not distribute upstream data-selection or feature-generation
scripts. Retained model-development/refitting scripts document the exact final
procedure and frozen configurations, but they are **method-reference scripts**,
not self-contained public entry points: full vector retraining/inference needs
the three-conformer xTB atom-graph store, vector-alignment table, and trained
checkpoints generated during the study. These raw intermediates are not needed
to reproduce the published evaluations, plots, source tables, or metrics.

The public data package provides the complete development matrix, the frozen 4,000-molecule internal test set, external validation tables, final prediction tables, and feature dictionary. See the repository root `README.md` for data locations and provenance.

## Public entry points

- `run_domain65k_candidate100_final_locked.py`: reproducible final evaluation on the released Test-4000 prediction table.
- `plot_external_rule_stratified_rank_validation.py`: external D validation for broad n=140 and its strict n=34 subset.
- `plot_p_rank_validation_panels.py`: Test-4000 and external P-rank validation panels.
- `plot_domain65k_consensus_interpretability.py`: Test-4000 ensemble-disagreement diagnostic.
- `plot_feature_relationship_heatmap.py`, `plot_domain65k_d_algorithm_comparison_4000.py`, and `plot_d_rank_density.py`: public Figure 2a–c plotting entry points.
- `plot_feature_interpretability.py`, `plot_chemical_space_coverage.py`, and `plot_candidate78_pareto_knee_selection.py`: public feature, coverage, and candidate-selection figures.
- `plot_frozen_fusion_validation.py`, `plot_vector_residual_summary.py`, and `plot_vector_training_dynamics.py`: source-data-driven supplementary panels.

## Method-reference scripts

`run_domain65k_d_v2_tabular_candidate100_fold0.py`,
`run_domain65k_d_vector_equivariant.py`,
`search_domain65k_candidate100_fold0_multiobjective_fusion.py`,
`refit_domain65k_d_candidate100_tabular.py`,
`refit_domain65k_d_candidate100_xgb.py`, and
`run_domain65k_p_candidate100_single_lgbm.py` document the selected training
and refitting procedures. See `frozen_configs/README.md` for final cohort
sizes and feature-manifest provenance.

## Python environment

For public evaluation and source-data plotting, install
`requirements-public.txt`. The original full workflow used Python 3.11 plus
`scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `torch`, `rdkit`, `h5py`,
`networkx`, `adjustText`, and `shap` for the corresponding method-reference
scripts.

## Public-release validation

`analysis_outputs/validate_public_release.py` checks the released Test-4000
predictions, nested external validation tables, frozen fusion weights,
candidate-selection table, and compact figure-source data. It recomputes the
reported rank metrics from public data and writes a concise JSON record under
`results/`.

The plotting scripts use `matplotlib`; the external-rank and consensus panels
also use `scipy` for rank statistics. Both are listed in
`requirements-public.txt`.

## Reproducing the published evaluations

After cloning the repository, run the following from the repository root:

```bash
python code/analysis_outputs/validate_public_release.py
python code/analysis_outputs/run_domain65k_candidate100_final_locked.py
python code/analysis_outputs/plot_external_rule_stratified_rank_validation.py
python code/analysis_outputs/plot_p_rank_validation_panels.py
python code/analysis_outputs/plot_domain65k_consensus_interpretability.py
python code/analysis_outputs/plot_feature_relationship_heatmap.py
python code/analysis_outputs/plot_domain65k_d_algorithm_comparison_4000.py
python code/analysis_outputs/plot_d_rank_density.py
python code/analysis_outputs/plot_feature_interpretability.py
python code/analysis_outputs/plot_chemical_space_coverage.py
python code/analysis_outputs/plot_candidate78_pareto_knee_selection.py
python code/analysis_outputs/plot_frozen_fusion_validation.py
python code/analysis_outputs/plot_vector_residual_summary.py
python code/analysis_outputs/plot_vector_training_dynamics.py
```

Metrics are written under `results/`; figure scripts write vector and raster outputs under `results/figures/`. These public entry points use only released prediction tables and do not alter the frozen models or published data.

Outputs are written under `results/`, which is ignored by Git. Figure source
tables and their panel mapping are documented in `data/figure_source/README.md`
and `metadata/figure_source_index.csv`.
