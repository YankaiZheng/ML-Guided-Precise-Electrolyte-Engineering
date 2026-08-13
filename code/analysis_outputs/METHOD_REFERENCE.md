# Method-reference scripts

The scripts in this directory fall into two different categories.

## Publicly runnable evaluation and plotting

The entry points listed in [`../README.md`](../README.md) operate only on
tables released with this repository. They reproduce the reported Test-4000
and external metrics and render the corresponding ML panels from public source
data.

## Retained study-protocol scripts

The remaining scripts document the original model-selection, refitting, and
atom-graph workflow. They are retained so that model choices, feature
selection, fusion, and the vector residual-learning protocol can be audited.
They are not advertised as standalone reproduction commands because the study
used an internal three-conformer xTB atom-graph store, vector-alignment tables,
and fitted binary artifacts that are not bundled in this release.

These scripts retain their original study-local paths on purpose: they are
archival protocol records, not executable paths in the public package.

In particular, `build_domain65k_tabular_shap_inputs.py` and
`compute_domain65k_weighted_treeshap.py` generated the original TreeSHAP
intermediates. Their released numerical outputs are under
`data/figure_source/`, and `plot_feature_interpretability.py` is the public
plotting route. Historical exploratory scripts, including
`run_domain65k_d_locked_test.py` and `plot_domain65k_design_rules_novelty.py`,
are not public manuscript entry points and must not be used to replace the
frozen Test-4000 metrics or figure source data.
