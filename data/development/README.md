# Development matrix

The complete Candidate100-clean development matrix is distributed as the
release asset `main_dataset.csv.gz` because it exceeds GitHub's normal Git-file
size limit. Download it from [Release v1.0.0](https://github.com/YankaiZheng/ML-Guided-Precise-Electrolyte-Engineering/releases/tag/v1.0.0)
and place it in this directory when a local copy is needed.

The table contains 60,641 records and 3,562 columns. It includes identifiers,
the prespecified `cv_fold`, D (debye), P (atomic units), and the complete
multiscale tabular feature matrix. The frozen Test-4000 feature matrix is a
separate release asset; its final prediction table is tracked at
`../test/test4000_predictions.csv`.

The final development and test cohorts are published as frozen analysis
resources. Upstream rule-filtering and feature-generation scripts are outside
the release scope; the released matrix is sufficient to inspect every final
tabular input column listed in `metadata/feature_dictionary.csv`.
