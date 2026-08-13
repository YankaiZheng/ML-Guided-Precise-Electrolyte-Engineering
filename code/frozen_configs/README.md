# Frozen final-model configuration

These manifests record the configurations frozen before Test-4000 evaluation.

| Component | Final refit cohort | Frozen input definition |
| --- | ---: | --- |
| D tabular (five members) | 60,641 Candidate100-clean records | 1,221-feature manifest |
| D XGBoost (two members) | 60,641 Candidate100-clean records | 1,093-feature historical fold-0 manifest |
| D equivariant vector | 60,554 graph-valid Candidate100-clean records | Three-conformer xTB atom graphs; 35 epochs at 3e-4 then one epoch at 5e-5 |
| P LightGBM | 60,641 Candidate100-clean records | 512-feature fold-0 training-only selection |

`d_xgboost_1093_feature_manifest.json` retains the historical `n_cv = 60,672`
field because that file is the original fold-0 feature-selection record. It
does **not** describe the final refit size. The final two XGBoost members were
refit on the 60,641 Candidate100-clean development records using this frozen
feature list. This distinction is intentional and avoids silently rewriting
the provenance of the selected feature list.

`final_d_fusion_weights.json` has eight member slots. `cat_rank` remains in
the frozen member list with a zero weight; six tabular members and the vector
branch have non-zero weights.
