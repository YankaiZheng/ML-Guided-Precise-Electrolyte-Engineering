# ML-Guided Precise Electrolyte Engineering

This repository hosts the data resources for the D-P Compass workflow used for molecular ranking in electrolyte-relevant chemical space. It contains the frozen internal test set, external validation sets, final prediction tables, a complete feature dictionary, and a versioned release asset for the 60,641-molecule development set.

## Data at a glance

| Resource | Location | Molecules | Contents |
| --- | --- | ---: | --- |
| Development set | GitHub Release `v1.0.0` | 60,641 | Candidate100-clean development records, D/P labels, identifiers, and complete multiscale feature matrix |
| Internal test features | `data/test/test4000_features.csv.gz` | 4,000 | Frozen complete-case test records, D/P labels, and complete multiscale feature matrix |
| Internal test predictions | `data/test/test4000_predictions.csv` | 4,000 | Frozen eight-member D predictions, final D fusion, and final P predictions |
| Broad external validation set | `data/external/external_broad_n140.csv` | 140 | External D/P labels and final-model member/fusion predictions |
| Strict external validation subset | `data/external/external_strict_n34.csv` | 34 | Strict Electrolyte-65K rule-aligned subset of the broad external set |
| Feature dictionary | `metadata/feature_dictionary.csv` | 3,562 variables | Feature provenance, units, missing-value encoding, and final-model selection flags |

The `n=34` external set is a strict subset of the `n=140` broad external set. It is not an independent second external test set.

## File descriptions

### Development data

`main_dataset.csv.gz` is provided as a GitHub Release asset rather than a Git-tracked file because its compressed size is approximately 584 MB, above GitHub's regular 100 MB file limit. The file is linked from the [v1.0.0 release](../../releases/tag/v1.0.0). Its SHA-256 checksum is recorded in `metadata/checksums.sha256`.

The development set contains the 60,641 Candidate100-clean records used for model development and final refitting. `cv_fold` identifies the prespecified development fold assignment. The fixed 4,000-molecule internal test set is not included in this file.

### Internal test data

`test4000_features.csv.gz` contains the frozen 4,000-molecule internal evaluation set and all complete tabular features. `test4000_predictions.csv` contains the corresponding labels and predictions from the final D and P models. The D target is recorded in debye and P is recorded in atomic units.

The test set was held out from the final model refitting and fusion-weight search. It is made public here to enable exact reproduction of reported test metrics and figures.

### External validation data

The external records use B3LYP/6-31+G(d,p) reference values, rather than the B3LYP/TZVP labels used for the QMe14S-derived internal data. The broad and strict tables include labels, member predictions, and the final eight-member D fusion output.

### Feature metadata

`metadata/feature_dictionary.csv` defines all columns in the complete feature matrix. The Boolean columns `selected_final_D_tabular`, `selected_final_D_xgboost`, and `selected_final_P_lightgbm` identify features used by the corresponding frozen final models.

## Provenance and attribution

Electrolyte-65K was derived by applying predefined electrolyte-relevant structural rules to QMe14S equilibrium-structure records, followed by complete-feature filtering and the frozen split protocol used in this study. QMe14S comprises 186,102 small organic molecules calculated at the B3LYP/TZVP level and is distributed under CC BY. Please cite both the original QMe14S publication and this repository when reusing these derived resources.

Yuan, M.; Zou, Z.; Hu, W. *QMe14S: A Comprehensive and Efficient Spectral Data Set for Small Organic Molecules.* **J. Phys. Chem. Lett.** 2025. https://doi.org/10.1021/acs.jpclett.5c00839

## Integrity checks

SHA-256 digests for every distributed data object are provided in `metadata/checksums.sha256`; row and column counts are listed in `metadata/data_manifest.csv`. For a local integrity check:

```bash
shasum -a 256 -c metadata/checksums.sha256
```

## License

The derived data and repository documentation are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse of upstream QMe14S content remains subject to its original CC BY attribution requirements.

## Citation

Please use `CITATION.cff` to cite this version of the data package. A permanent archival DOI should be added after the first Zenodo archival release.
