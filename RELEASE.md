# Release Notes

This file is the single place for versioned release notes, wheel version
references, and install targets.

## Current Release

- Version: `0.1.6`
- Release date: `2026-07-09`
- Package name: `Combo2`
- Protected wheel: `dist_protected/combo2-0.1.6-cp313-cp313-linux_x86_64.whl`
- Python target: `3.13`
- Installed target in this workspace: `python3` (`Python 3.13.11`), package `Combo2 0.1.6`

Install the current wheel:

```bash
python -m pip install dist_protected/combo2-0.1.6-cp313-cp313-linux_x86_64.whl
```

## 0.1.6

Changes:

- Changed the local ZZ500 benchmark helper from adjusted `pre_close` to raw `real_pre_close` so benchmark plots and summaries follow a price-index style series instead of a reinvested total-return style series.

## 0.1.5

Changes:

- Refactored feature data around explicit frequency groups: supported `freq` values are `1d`, `5m`, and `1m`; missing `freq` defaults to `1d`; labels only support `1d`.
- Preserved intraday cube factors end-to-end: `1d` data stays `[date, code]`, `5m` data stays `[date, 49, code]`, and `1m` data stays `[date, 239, code]` with strict canonical bar validation.
- Added `FeatureGroups` and `GroupCodec`, exposed `FeatureGroups` from `comb2`, and changed `ComboDataLoader`, `ComboTrainDataset`, `ComboBuffer`, and `ComboBase` to pass grouped tensors to `ResearchModel`.
- Injected `freqs` and `num_features_by_freq` into model config while retaining `num_features` as the total feature count.
- Extended `DataRegistry.get_data(name, start_ds, end_ds)` as the public range-loading reader for declared factor, label, and aux items.
- Removed `processed_feature_cache` from runtime config and training flow; grouped feature tensors are encoded directly through the configured codec.
- Removed `nbar`, reducer ops, and reducer helper functions from the data-layer interface. Item ops are limited to the explicit whitelist: `cs_zscore`, `zscore`, `rank`, `truncate`, `nan_to_num`, `fillna`, `winsorize_by_quantile`, `normalize_by_max_abs`, `rolling_mean`, `rolling_std`, and `neut`.
- Limited `neut` to `1d` data and kept rolling ops on the date axis.
- Changed current-time handling so `current_ti` masks only future intraday bars on the final day of train/predict windows; historical days remain complete.
- Updated FP4 codec handling so non-finite values use the reserved code instead of failing during quantization.
- Removed the `runAblationByZero.py` CLI and its CLI tests.
- Updated `comboHelloWorld`, `eg-lgbm`, `eg-torch`, root docs, package docs, architecture docs, and `config.human` to document `FeatureGroups`, cube/freq rules, and researcher override hooks.
- Added and updated tests for cube feature groups, grouped codec behavior, config validation, fixed bar counts, future-bar masking, and public exports.

## 0.1.4

- Published distribution name is now `Combo2`; wheel filenames are normalized to lowercase `combo2-...`.
- Unified release/version references around `Combo2 0.1.4` in `README.md`, `packaging/README.md`, `packaging/build_protected_wheel.py`, and this file.
- Reworked config/data path handling to prefer `builtin.factorsim` and explicit item paths instead of `constants.factor_root`.
- Added runtime config support for `snap_ti`, `seed`, and `deterministic`, and wired runtime seed initialization into `ComboBase`.
- Refactored 3D `builtin.factorsim` loading to an ops-only pipeline: `nbar` is pre-window selection, cube aggregation is expressed through item ops, and the final pipeline must end as `[date, code]`.
- Extended item ops with named-axis reducers and rolling ops: `rank`, `last`, `mean`, `std`, `sum`, `max`, `min`, `rolling_mean`, `rolling_std`, plus axis-aware `winsorize_by_quantile` and `normalize_by_max_abs`.
- Bound processed feature cache to `(ds, ti)` and propagate current `ti` through loader/registry so intraday snapshots produce distinct cached features.
- Clarified that processed feature cache stores post-processed daily features only; raw 3D minute cubes are not kept as a persistent reader-side source cache.
- Changed `runEval --sim/--pnl/--corr/--va/--exposure` to direct local parquet/csv modes; these single-item modes no longer accept `config.xml`.
- Changed `runEval --corr` and `comb-eval matrix-corr` to default to the most recent 240 overlapping days.
- Kept checkpoint reuse behavior simple: skip retraining on exact-date checkpoint hits; otherwise retrain only when the latest checkpoint is more than 30 trading days stale.
- Sized processed feature cache during training from the effective train window plus `retDays` and `trainDelay`.
- Updated starter/example configs, researcher docs, architecture docs, and tests to match the new `builtin.factorsim`, runtime, and `runEval` behavior.
