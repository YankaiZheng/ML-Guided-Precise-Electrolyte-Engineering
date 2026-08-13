# Public data layout

This directory contains the source data required to inspect the reported
D-P Compass results. The two complete feature matrices are supplied as GitHub
Release assets; all final labels, predictions, external-validation tables,
candidate rankings, and compact figure-source tables are tracked in the
repository.

| Resource | Location | Rows | Purpose |
| --- | --- | ---: | --- |
| Candidate100-clean development matrix | `development/main_dataset.csv.gz` release asset | 60,641 | Final development/refit cohort and complete multiscale tabular matrix |
| Test-4000 feature matrix | `test/test4000_features.csv.gz` release asset | 4,000 | Frozen internal evaluation feature matrix |
| Test-4000 labels and predictions | `test/test4000_predictions.csv` | 4,000 | Exact internal D and P evaluation, member outputs, and final scores |
| Broad external validation | `external/external_broad_n140.csv` | 140 | External B3LYP/6-31+G(d,p) D/P references and frozen predictions |
| Strict external subset | `external/external_strict_n34.csv` | 34 | Predefined strict Electrolyte-65K-rule subset of the broad cohort |
| Final electrolyte candidates | `candidates/` | 78 | Frozen D/P predictions and Pareto-knee selection inputs |
| Figure source data | `figure_source/` | varies | Compact tables underlying the ML figures and supplementary panels |

## Fixed internal test set

`test4000_predictions.csv` is the formal paper evaluation table. It contains
4,000 preserved records and 3,998 unique canonical SMILES because two source
identities occur twice in the underlying QMe14S-derived records. Reported
metrics are record-level metrics on the frozen 4,000 rows. The table was not
used to select final fusion weights or refit final models.

The test records were fixed before final refitting by the study's stricter
chemical-applicability split protocol; the remaining complete-feature records
form the Candidate100-clean development cohort. The frozen record identifiers
and all feature values are released here so that the reported evaluation does
not depend on an undisclosed subset.

The formal D prediction is `score__final_fusion_multiobjective`; the formal P
prediction is `P_pred__single_lgbm_raw`. `score__final_fusion` is a retained
legacy audit column and is not used for reported results.

## External validation

The external tables use B3LYP/6-31+G(d,p), also written B3LYP/6-31+G**, rather
than the B3LYP/TZVP references of the QMe14S-derived internal data. The strict
n=34 table is nested in the broad n=140 table, not a second independent test
set. For each reported external cohort, the frozen eight-member weights are
applied after member outputs are rank-normalized within that cohort. The
resulting formal column is `D_score__recomputed_subset`.

## Figure source data

See `figure_source/README.md` and `metadata/figure_source_index.csv`. The
published source tables are compact plotting inputs and result tables; they do
not require the unpublished raw xTB atom-graph store to inspect the figures.
