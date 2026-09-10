# Release Notes

The repository-root `VERSION` file is the single source of truth for the wheel
version. This file records release notes, published artifacts, and install targets.

## Current Version

- Version: `1.0.2`
- Version date: `2026-09-09`
- Package name: `combo2`
- Protected wheel build target: `dist_protected/combo2-1.0.2-cp313-cp313-linux_x86_64.whl`
- Python target: `3.13`
- Status: the `1.0.2` wheel was built and installed locally; it has not been published to a remote package registry.

Build and install this version:

```bash
python packaging/build_protected_wheel.py --python python3.13
python -m pip install dist_protected/combo2-1.0.2-cp313-cp313-linux_x86_64.whl
```

## Unreleased

No unreleased changes.

## 1.0.2

Bug fixes:

- Apply the default `BaseUnivMask`, `NoNewStockMask`, and `LimitMask` intersection consistently to training, signals, and IC evaluation when `cache_path` is configured. Explicit researcher validity sources continue to replace that default.
- Preload training feature windows and targets once per dataset, reuse rolling prediction windows across dates and sample times, and refresh only the current live source point after a cache update. Feature compression remains limited to feature storage.
- Read snapshot targets with their declared delay and prefetch them in bounded ranges so evaluation reuses the same target and validity arrays for IC and decile calculations.
- Align two-dimensional factor sources to the configured instrument axis, preserving missing instruments as `NaN` and source precision. Normalize infinite feature values as missing before preprocessing.
- Build execution buyable and market-sellable pools from valid prices and market masks. Freeze non-tradable holdings at their actual amount, reject orders outside the corresponding pool, and write off confirmed delistings at zero with settlement records and daily metrics.
- Preserve the configured optimizer benchmark independently from the `ZZ500` universe constraints and use actual stock-book holdings for `opt1` turnover normalization.
- Validation covers 116 passing regression tests and 11 conditional skips.

## 1.0.0

Research interfaces:

- Declare ordinary Memmap sources in `ResearchLoader.data_requirements()`. Source paths are relative to the defining Python file; each source has an explicit nonnegative trading-day `delay`, applied once during reading.
- Use `process_source(source, loaded)` for sampling-time selection, dimensionality reduction, and derived feature columns. Its result is `[date, stock, feature]`; raw minute/bar arrays are released after processing.
- Select model inputs, target fields, and validity fields in `model_input_sources()`, `model_target()`, and `model_validity_source()`. Researchers index label/returns fields in the loader and remain responsible for target observability at the configured training cutoff.
- Use `runtime.sample_times` for both single-time and intraday runs. A history window contains the same sampling time across `tsDays` trading days. Dataset samples are `(idx, ds, ti, x, y, w)`, and `predict(x_window, di=..., ti=...)` receives a plain `[tsDays, stock, feature]` tensor.
- Replace XML data declarations, `role`, frequency-grouped inputs, `FeatureGroups`, and `GroupCodec` with the Python source and array interfaces. Reading settings belong under `<combo><loader>`. Existing research projects must update their declarations, sample unpacking, and prediction signatures; interface examples are in [config.human](config.human).
- Retain training scheduling, `trainDelay`, checkpoint IO, and model smoothing. Feature windows are built on demand using bounded caches and chunked prefetch; FP4/FP8 codecs operate on tensors without quantizing source prices or targets. Torch and LightGBM examples preserve their declared factor lists in Python.

Execution and evaluation:

- Set `backtest.execution_price="source:column"` to select a researcher-provided execution price. Multiple intraday sample times require `opt2`; execution-window consistency is owned by the researcher.
- Preserve cash, actual holdings, T+1 locked shares, the daily budget, and cumulative executed turnover across intraday steps. Unlock and reset daily state only when the trading day changes, and settle daily PnL once at the final configured time.
- Record timestamped orders and fills in `executions.csv`, timestamped target positions, and daily execution metrics. Include the initial trading day's PnL and fees in performance summaries.
- Evaluate `(date, time)` alpha with the research loader's raw target and validity definitions. Report per-time IC, target deciles, Barra/CAP exposures, and PnL from actual execution records. Configuration evaluation no longer accepts separate CLI label overrides.
- Preserve sampling times when reading CSV/Parquet matrices for correlation and exposure; correlation lookbacks continue to count trading days.
- Added `strategy.optimizer.benchmark`, defaulting to `000905.SH`, so optimizer risk, industry, and relative-variance calculations can use a configured index without changing the independent `ZZ500` universe constraints or backtest reporting benchmark.

Integration and fixes:

- Handle partially written intraday Memmap indexes under pandas 3 and refresh live source/feature caches at the current point. Skip invalid or nonpositive execution prices.
- Resolve researcher loader/dataset paths and cache paths when Optuna renders trial configs. Exclude generated build directories from package discovery during repeated protected-wheel builds.
- Update the starter, examples, researcher documentation, evaluation commands, and package exports together. Source validation includes 104 passing regression tests (10 conditional skips), plus a real two-day/four-time Torch–MOSEK execution and evaluation pipeline, T+1 and exhausted-turnover checks, live/historical prediction consistency, and protected-wheel installation checks performed before the version bump.

## 0.1.10

Changes:

- Added `constants.freq` as the execution-mode switch with `1d` as the default. Daily execution keeps the original factor/label dataset, `predict(x_window)`, daily IC, and backtest contract; `5m`/`1m` execution requires exactly one same-frequency returns target and uses `(di, ti)` samples and predictions.
- Added causal intraday window clipping, per-target-time alpha and IC outputs, and explicit model config fields for `freq`, `target_freq`, and `target_times`. Intraday execution does not invoke the daily backtest or `runEval` workflow.
- Kept the processed data registry as a bounded 64-day per-item LRU and synchronized the starter, examples, researcher guide, root/package documentation, architecture guide, Optuna scope, and evaluation scope with the dual execution contracts.
- Restored the original daily `retDays` and optional `snap_ti` label path while keeping snapshot labels, alpha IC, configured evaluation, and backtest execution prices on the same `IntraVwap.Vwap30.HHMMSS` source.
- Replaced the default median long-only strategy with the MOSEK Fusion optimizer modeled after the reference holding optimizer. Alpha is split at raw zero and normalized separately by sign; the optimizer uses actual realized stock-book holdings for turnover and sell-only candidates.
- Added `opt1` and `opt2` optimizer modes with amount-order output. `opt2` applies T+1 sellability, shared intraday turnover budgets, direct amount-based liquidity constraints, and zero-order observable fallback behavior when no optimal solution is available.
- Added XML-configurable optimizer parameters with delay-1 defaults, ZZ500 weights and BarraCNE5 inputs under `constants.cache_path/AshareCache`, and pinned `Mosek==11.0.25` wheel dependency metadata. The repository includes the approved sanitized `mosek.lic`; protected wheels do not embed it, so deployments select an authorized copy through `MOSEKLM_LICENSE_FILE`.
- Added real Cache and MOSEK coverage for one-day solving and two-day actual-holding turnover behavior, plus configuration and normalization contract tests.
- Added CAP correlation to full configuration evaluation. It aligns alpha with the market cap available on the same trading day, cross-sectionally ranks market cap, and correlates it with median-centered, separately normalized long/short alpha weights.
- Added `cap_corr_summary.csv` with annual and all-sample `cap_corr.avg`, `cap_corr.ir`, and `cap_corr.std` values. CAP correlation is also included in CLI text output and the signal-analysis summary panel.
- Extended `--skip-exposure` to skip both Barra-style exposure and CAP correlation. The report continues when either cache-backed exposure calculation is unavailable and records the reason in its messages.
- Added focused CAP-correlation tests for same-day market-cap alignment and all-sample aggregation, and updated evaluation documentation and CLI coverage for the layered-IC public names.

## 0.1.9

Changes:

- Finalized the public layered-IC names: daily files now write `lic` and `layerspread`; normalized summaries expose `lIC.avg`, `lIC.ir`, and `layerSpread.avg`; the report and signal-summary chart label the IR as `lIR`.
- Removed percentile IC (`percic`) from daily metric generation, normalized summaries, report key columns, charts, CLI output, and L1/L2 evaluation rules. Existing `percic`/`percIC` columns are ignored when an IC file is summarized.
- Updated `runEval --sim --normalize-names` to map `lic` to `lIC` and `layerspread` to `layerSpread`. It also accepts the prior normalized `layeric` column name as `lIC`; downstream consumers should replace any dependence on `percic` with the layered metrics.
- Centralized daily IC calculation so the simulation tool and full evaluation report produce the same `ic`, `5dic`, `rankic`, `lic`, `layerspread`, and coverage fields.
- Kept the existing L1/L2 cutoff values while moving the layered check to `lIC.avg`; these thresholds should be recalibrated against historical layered-IC distributions before being treated as a new baseline.

## 0.1.8

Changes:

- Introduced layered IC as the replacement for percentile IC in overall evaluation. On each date, valid alpha and 1-day forward-return pairs are placed into ten equal-frequency alpha layers (Q1--Q10); the metric is the correlation between layer number and each layer's mean return.
- Added the daily layered IC, its cross-date mean and information ratio, and the `layerSpread` Q10--Q1 mean-return difference to IC summaries and the signal-analysis report.
- Preserved a literal Q10--Q1 interpretation: tied alpha values are never split between layers, and dates that cannot form all ten non-empty layers are excluded from layered-IC and spread aggregation.

## 0.1.7

Changes:

- Consolidated the framework implementation under the single `comb2` import package and removed the generic top-level `src` package from source and protected-wheel builds.
- Made `isTrainDay` the sole training-calendar rule. Checkpoint age no longer suppresses or triggers training; only an exact target-date checkpoint skips an otherwise scheduled training run.
- Fixed the published distribution name to `combo2`, removed build-time name overrides, and made the repository-root `VERSION` file the only build version source. CI reads the same identity when building the protected wheel.
- Consolidated runtime artifacts under `constants.output_root`. `train.log`, `alpha_history.pt`, `alpha.parquet`, checkpoints, backtest results, and evaluation reports now use fixed derived paths; redundant output and checkpoint path settings were removed from configs, examples, Optuna rendering, and documentation.
- The complete `barra` preset maps all 11 logical style names to the canonical uppercase `BarraCNE5.*` filenames derived from `constants.cache_path`.
- Clarified training-date semantics and changed starter configs to `trainDelay=0`: no hidden framework offset is added, because labels retain their stored delay convention. `retDays` continues to align each factor date with the future-return window beginning on that date, while `tsDays` only controls the trailing feature window.
- Added config validation for training windows, thread counts, data offsets, smoothing, strategy dates, and backtest constraints. Negative `trainDelay`, non-positive `retDays`/`tsDays`, and training windows shorter than `tsDays` now fail during config loading.
- Removed the selection-module API, `select_days`, `<combo><defaults>`, selection hooks, and package exports. Training-window calculation is now explicit in `ComboBase`, bounded directly by `trainDelay`, `retDays`, `tsDays`, and `max_train_days`.
- Centralized the AshareCache layout under `constants.cache_path`. Labels, stock masks, Barra data, backtests, and evaluation all use the same parent-root derivation; per-loader `ashare_data_path` and custom mask-path settings were removed, with `StockMask2.NoNewStockMask`, `StockMask2.LimitMask`, and `StockMask2.BaseUnivMask` as the standard universe inputs. `runEval --exposure` now requires `--cache-path` with that parent directory.
- Unified alpha IC calculation with the report implementation. Signals and 1-day/5-day labels are aligned on common dates and instruments and filtered by the shifted intersection of `BaseUnivMask` and `LimitMask`, so `runCombo` and evaluation reports use the same metrics and universe rules.
- Corrected the example Torch IC loss to compute a masked, weighted Pearson correlation with valid centering and normalization; zero-weight instruments no longer affect the loss.
- Changed `runCombo` to tee standard output and errors to the derived `train.log` while preserving console output.
- Updated generated projects and Torch examples to use the public `FeatureGroups` interface and refreshed root, package, packaging, and configuration documentation for the new paths and removed APIs.
- Updated starter/example configurations to use relocatable factor paths and `constants.cache_path` rather than machine-specific absolute AshareCache paths.
- CI now only builds and publishes the protected wheel; pytest remains a local validation step so the build runner does not require pytest to be installed.

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
