# Release Notes

This file is the single place for versioned release notes, wheel version
references, and install targets.

## Current Release

- Version: `0.1.4`
- Release date: `2026-07-07`
- Package name: `Combo2`
- Protected wheel: `dist_protected/combo2-0.1.4-cp313-cp313-linux_x86_64.whl`
- Python target: `3.13`
- Installed target in this workspace: `python3` (`Python 3.13.11`), package `Combo2 0.1.4`

Install the current wheel:

```bash
python -m pip install dist_protected/combo2-0.1.4-cp313-cp313-linux_x86_64.whl
```

Current local build artifacts also include:

- `dist_protected_combo2_014/combo2-0.1.4-cp313-cp313-linux_x86_64.whl`

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
