# Figure source data

These CSV files are the compact numerical inputs underlying the ML figures and
supplementary panels. They are supplied independently of image files so that
reported values, rank coordinates, feature attributions, and candidate
selection can be inspected without access to the study's raw intermediate
xTB atom-graph store.

`../../metadata/figure_source_index.csv` maps each source table to its panel
and public plotting script. Tables headed `Fig_` preserve the source-data names
used when the original panels were exported. The files retain numerical
precision; rounding belongs only in manuscript display text.

The molecular drawings in the vector-correction panel are visual annotations,
whereas `fig3_representative_atomic_corrections.csv` provides the underlying
atom-level correction magnitudes. The full three-conformer atom-graph store is
not required to inspect the reported correction values or reproduce the
source-data-driven panels.
