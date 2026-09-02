# comb2 Architecture

This document describes internal component boundaries. Researcher-facing configuration and hook usage live in `../../../config.human`.

## DataRegistry

`comb2/DataRegistry.py` owns data declaration, loading, alignment, item-level ops, and processed data cache.

Main types:

| Type | Responsibility |
|---|---|
| `Universe` | Shared axis metadata: `dates`, `codes`, `dtype`, `date2idx()`, `idx2date()`, `code2idx()`. |
| `DataItem` | One declared data source from config: `name`, `module`, `path`, `role`, `ops`, `params`. |
| `OpSpec` | One data item op declaration. |
| `DataRegistry` | Maintains processed data tensors, module cache, load state, and `get_data(name, start_ds, end_ds)`. |

Data cache model:

```text
processed_cache[name]    # OrderedDict[date_idx, processed row]
cache_days               # per-item LRU day limit, default 64
module_cache[...]        # reader objects / module-local helpers
```

The registry cache is bounded per item. It does not allocate a tensor covering the complete universe date axis and does not maintain a `processed_loaded` bitmap.

Data load flow:

```text
DataRegistry._ensure_range(names, start_ds, end_ds)
  -> _ensure_processed_range(name, start_ds, end_ds)
     -> resolve op requirements: dependencies and lookback
     -> _ensure_processed_range(dep, ...)
     -> module function(item, registry, start_ds, end_ds)
     -> validate returned shape by item freq
        1d: [R, N]
        5m: [R, 49, N]
        1m: [R, 239, N]
     -> _apply_ops(item, loaded_window, lo_idx, hi_idx)
     -> write each requested day to the per-item LRU

DataRegistry.get_data(name, start_ds, end_ds)
  -> ensures the requested date range is loaded
  -> rejects ranges longer than cache_days
  -> stacks and returns the requested cached rows

DataRegistry.get_data(name)
  -> returns the current cache only when it is non-empty and contiguous
  -> otherwise requires an explicit date range
```

Supported built-in modules:

| Module | Responsibility |
|---|---|
| `builtin.factorsim` | Default reader for factor, label, Barra style, and AshareCache memmap data. `1d` sources are `[date, code]`; `5m` and `1m` sources are strict canonical cubes `[date, bar, code]`. |
| `builtin.alpha_parquet` | Read alpha parquet and align columns to `Universe.codes`. |
| `my_data:load_data` | Project custom reader. |

Supported item ops:

| Op | Responsibility |
|---|---|
| `cs_zscore` / `zscore` | Zscore on the requested axis. |
| `rank` | Rank on the requested axis; optional `pct` keeps percentile rank semantics. |
| `truncate` | Clamp by `min` and `max`. |
| `nan_to_num` / `fillna` | Fill NaN/inf with `value`. |
| `winsorize_by_quantile` | Quantile winsorization on the requested axis. |
| `normalize_by_max_abs` | Max-abs normalization on the requested axis. |
| `rolling_mean` / `rolling_std` | Rolling ops over the `date` axis. |
| `neut(name, ..., ratio)` | Neutralize `1d` data against one or more `1d` dependencies; optional final `ratio` defaults to `1.0` and may be a per-dependency ratio array. |

Item ops are limited to the whitelist above. `last/mean/std/sum/max/min` and `nbar` are not part of the current data-layer interface; intraday aggregation belongs in `ResearchLoader` or `ResearchModel`.

## ComboDataLoader

`comb2/DataLoader.py` owns model feature/label generation, prediction windows, masks, and cache orchestration.

Main types:

| Type | Responsibility |
|---|---|
| `LoaderConfig` | Runtime data loader settings parsed from config. |
| `ComboDataLoader` | Generates single-day feature, label, prediction window, and valid masks. |
| `ComboTrainDataset` | Builds rolling training tensors `X/Y/W` from a loader. |
| `ComboBuffer` | Stores recent prediction features for online/history combination. |

Execution modes:

| `constants.freq` | Supervision | Sample and prediction contract |
|---|---|---|
| `1d` | Zero or one `role="label"`; training requires a label | `(idx, x, y, w)` and `predict(x_window)` |
| `5m` / `1m` | Exactly one same-frequency `role="target"` | `(idx, di, ti, x, y, w)` and `predict(x_window, di=..., ti=...)` |

The loader receives `constants.freq` as `LoaderConfig.freq`. Daily input items are `role="factor"`; in intraday execution every non-target item is a model input.

Feature flow:

```text
ComboDataLoader.gen_feature(ds)
  -> align_date(ds)
  -> _build_feature(ds)
     -> build_raw_feature(ds)
        -> source_date(ds, freq)
           daily execution: ds
           intraday execution with a 1d input: previous trading day
        -> registry.get_data(name, source_ds, source_ds)[0]
        -> group model inputs by freq
           1d: [N, F_1d]
           5m/1m: [N, bar, F_freq]
        -> inf -> NaN
     -> preprocess_features(groups, ds)
        -> preprocess_feature_group(freq, group, ds)
        -> default 1d: cs_zscore/truncate/nan_to_num
        -> default intraday: no-op
  -> return FeatureGroups
```

Feature cache note:

- Registry cache stores at most `cache_days` processed rows per item while preserving item shape.
- `ComboTrainDataset.X` and `ComboBuffer` store grouped tensors through `GroupCodec`.
- `current_ti` is used only by daily snapshot execution. Intraday execution passes the target `ti` explicitly when transforming each train/predict window.

Label flow:

```text
ComboDataLoader.gen_label(ds, ret_days)
  -> align_date(ds)
  -> label cache lookup
  -> resolve ret_days date range
  -> registry._ensure_range(label_name, start_ds, end_ds)
  -> read daily label rows from registry.get_data(label_name)
  -> nan_to_num(daily_returns, 0)
  -> weighted aggregate ret_days returns
  -> valid_mask = gen_valid_mask(start_ds) & finite(aggregated_label)
  -> preprocess_label(label_values, valid_mask, ds, ret_days)
     -> winsorize valid values
     -> demean
     -> std scale
     -> truncate(-3, 3)
     -> normalize_by_max_abs
     -> invalid fill 0
  -> write label cache
  -> return (y, w)
```

Intraday target flow:

```text
ComboDataLoader.gen_target(ds, ti)
  -> gen_raw_target(ds, ti)
     -> validate ti on the target frequency axis
     -> registry.get_data(target_name, ds, ds)[0][target_bar_id]
  -> gen_valid_mask(ds), using the previous trading day's masks
  -> finite(target) intersection
  -> preprocess_target(target_values, valid_mask, ds)
  -> return (y, w)
```

Prediction/live feature window flow:

```text
ComboBase.GenComboPos(ds, ti)
  -> buffer_load(ds)
     -> repeated gen_feature(ds) into ComboBuffer
  -> ComboBuffer.get(trailing tsDays)
  -> optional model trainii stock selection
  -> transform_feature_window(...)
     daily: stage="predict"
     intraday: target_ti=ti, stage="predict"
  -> ResearchModel.predict(...)
  -> refill predictions to the complete stock axis
```

For intraday execution, `transform_feature_window` retains only source bars whose completion progress is no later than the preceding target bar. The special opening-auction target bar is excluded from training, prediction, IC, and output.

## ResearchLoader Hooks

`ResearchLoader` must inherit `ComboDataLoader`. Hooks are layered; overriding a parent organization function bypasses its child hooks unless the new implementation calls them.

```text
gen_feature(ds)
  -> build_raw_feature(ds)
  -> preprocess_features(features, ds)
     -> preprocess_feature_group(freq, feature, ds)

gen_label(ds, ret_days)
  -> load label data
  -> aggregate ret_days returns
  -> build valid_mask
  -> preprocess_label(label_values, valid_mask, ds, ret_days)

gen_target(ds, ti)
  -> load one target bar
  -> build valid_mask
  -> preprocess_target(target_values, valid_mask, ds)
```

Recommended override level:

| Goal | Hook |
|---|---|
| Replace zscore with rank, or change one frequency's feature fill/truncate | `preprocess_feature_group(freq, feature, ds)` |
| Change the default daily preprocessing only | `preprocess_daily_features(feature, ds)` |
| Change raw feature assembly or raw derived features | `build_raw_feature(ds)` |
| Replace complete feature generation | `gen_feature(ds)` |
| Change label standardization, clipping, or sample weights | `preprocess_label(label_values, valid_mask, ds, ret_days)` |
| Replace complete label source/aggregation/valid-mask logic | `gen_label(ds, ret_days)` |
| Change intraday target standardization or sample weights | `preprocess_target(target_values, valid_mask, ds)` |
| Replace complete intraday target extraction | `gen_target(ds, ti)` |
| Change train/predict window processing while preserving causality | `transform_feature_window(feature_window, target_ti=..., stage=...)` |
| Change prediction stock availability | `_feature_available_mask(feature_window)` |
| Change full prediction window processing | `process_feature_window(feature_window)` or `load_feature_window(end_ds, ts_days)` |

`preprocess_features(features, ds)` is the framework-level dispatcher over all configured frequency groups. Research code should usually override `preprocess_feature_group(freq, feature, ds)` or `preprocess_daily_features(feature, ds)` instead. If `preprocess_features` is overridden, it must preserve the `FeatureGroups` contract: only configured freq keys exist, missing freqs must stay absent, and each returned group must keep the same stock axis and frequency semantics.

## ComboTrainDataset

Daily training dataset construction:

```text
ComboTrainDataset.__init__
  -> align end date and train window
  -> _build_validinsts()
  -> allocate X/Y/W
  -> chunked prefetch_features / prefetch_labels
  -> repeated loader.gen_feature(feature_ds)
  -> repeated loader.gen_label(label_ds, ret_days=x_delay)
  -> write X/Y/W

ComboTrainDataset.__getitem__(idx)
  -> decode X[idx:idx + ts_days]
  -> select y/w at last day of window
  -> return idx, x, y, w
```

Intraday training dataset construction:

```text
ComboTrainDataset.__init__
  -> build trailing feature storage ending before/at each target date
  -> chunked prefetch_features / prefetch_targets
  -> Y/W shape [day, target_bar, stock]

ComboTrainDataset.__getitem__(idx)
  -> map idx to (di, ti)
  -> decode the trailing ts_days feature window
  -> transform_feature_window(..., target_ti=ti, stage="train")
  -> select y/w for the same target bar
  -> return idx, di, ti, x, y, w
```

Research dataset override points:

| Goal | Hook |
|---|---|
| Change training stock universe | `_build_validinsts()` |
| Change sample return structure | `__getitem__(idx)` |

## Model Integration

`ComboBase` wires the runtime:

```text
config XML
  -> constants.freq and <combo><data> declarations
  -> LoaderConfig
  -> ResearchLoader or ComboDataLoader
  -> ResearchDataset or ComboTrainDataset
  -> ResearchModel.fit(dataset)
  -> GenComboPos builds a causal window through ComboBuffer
  -> daily: ResearchModel.predict(x_window)
     intraday: ResearchModel.predict(x_window, di=..., ti=...)
```

Daily model-facing shapes:

```text
Training sample: idx, x, y, w
x: FeatureGroups
x["1d"]: [ts_days, M, F_1d]
x["5m"]: [ts_days, M, 49, F_5m] if configured
x["1m"]: [ts_days, M, 239, F_1m] if configured
y: [M]
w: [M]

Prediction input:
x_window: FeatureGroups with the same grouped shapes
```

Intraday changes the sample header to `idx, di, ti, x, y, w`; grouped feature shapes remain the same, but the final day's intraday groups are causally clipped for `ti`. `ComboBase` injects `freq` in both modes and adds `target_freq` and `target_times` in intraday mode.

## runCombo Outputs

| Execution mode | Alpha index | Analysis and execution |
|---|---|---|
| `1d` | date | `daily_ic`, daily backtest, and `backtest/` outputs |
| `5m` / `1m` | `(dates, times)` | `intraday_ic.csv` and `ic_by_time.csv`; no daily backtest |
