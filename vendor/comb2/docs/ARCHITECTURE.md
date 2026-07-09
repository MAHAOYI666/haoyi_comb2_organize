# comb2 Architecture

This document describes internal component boundaries. Researcher-facing configuration and hook usage live in `../../config.human`.

## DataRegistry

`src/DataRegistry.py` owns data declaration, loading, alignment, item-level ops, and processed data cache.

Main types:

| Type | Responsibility |
|---|---|
| `Universe` | Shared axis metadata: `dates`, `codes`, `dtype`, `date2idx()`, `idx2date()`, `code2idx()`. |
| `DataItem` | One declared data source from config: `name`, `module`, `path`, `role`, `ops`, `params`. |
| `OpSpec` | One data item op declaration. |
| `DataRegistry` | Maintains processed data tensors, module cache, load state, and `get_data(name, start_ds, end_ds)`. |

Data cache model:

```text
processed_cache[name]    # aligned processed [T, N]
processed_loaded[name]   # processed date bitmap
module_cache[...]        # reader objects / module-local helpers
```

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
     -> write processed_cache

DataRegistry.get_data(name, start_ds, end_ds)
  -> ensures the requested date range is loaded
  -> returns processed_cache[resolved_name][lo:hi+1]

DataRegistry.get_data(name)
  -> returns already-loaded processed_cache[resolved_name]
  -> raises if the item has not been loaded yet
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

`src/DataLoader.py` owns model feature/label generation, prediction windows, masks, and cache orchestration.

Main types:

| Type | Responsibility |
|---|---|
| `LoaderConfig` | Runtime data loader settings parsed from config. |
| `ComboDataLoader` | Generates single-day feature, label, prediction window, and valid masks. |
| `ComboTrainDataset` | Builds rolling training tensors `X/Y/W` from a loader. |
| `ComboBuffer` | Stores recent prediction features for online/history combination. |

Feature flow:

```text
ComboDataLoader.gen_feature(ds)
  -> align_date(ds)
  -> _build_feature(ds)
     -> build_raw_feature(ds)
        -> registry._ensure_range(factor_names, ds, ds)
        -> registry.get_data(name)[date_idx] for role="factor"
        -> group factors by freq
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

- Registry cache stores processed tensors per item while preserving item shape.
- `ComboTrainDataset.X` and `ComboBuffer` store grouped tensors through `GroupCodec`.
- `current_ti` is not part of `gen_feature(ds)`; it masks only the last day of train/predict windows.

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

Prediction/live feature window flow:

```text
load_feature_window(end_ds, ts_days)
  -> choose trading days
  -> repeated gen_feature(ds)
  -> pad with NaN if history is short
  -> transform_feature_window(feature_window, stage="predict")
     -> mask future intraday bars on the last day only
  -> process_feature_window(feature_window)
     -> _feature_available_mask(feature_window)
     -> return selected FeatureGroups, available_mask
```

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
| Change train/predict window processing while preserving future-bar masking | `transform_feature_window(feature_window, stage=...)` |
| Change prediction stock availability | `_feature_available_mask(feature_window)` |
| Change full prediction window processing | `process_feature_window(feature_window)` or `load_feature_window(end_ds, ts_days)` |

`preprocess_features(features, ds)` is the framework-level dispatcher over all configured frequency groups. Research code should usually override `preprocess_feature_group(freq, feature, ds)` or `preprocess_daily_features(feature, ds)` instead. If `preprocess_features` is overridden, it must preserve the `FeatureGroups` contract: only configured freq keys exist, missing freqs must stay absent, and each returned group must keep the same stock axis and frequency semantics.

## ComboTrainDataset

Training dataset construction:

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

Research dataset override points:

| Goal | Hook |
|---|---|
| Change training stock universe | `_build_validinsts()` |
| Change sample return structure | `__getitem__(idx)` |

## Model Integration

`ComboBase` wires the runtime:

```text
config XML
  -> <combo><data> declarations and attrs
  -> LoaderConfig
  -> ResearchLoader or ComboDataLoader
  -> ResearchDataset or ComboTrainDataset
  -> ResearchModel.fit(dataset)
  -> GenComboPos uses loader.load_feature_window(...)
  -> ResearchModel.predict(x_window)
```

Default model-facing shapes:

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
available_mask: [N]
```
