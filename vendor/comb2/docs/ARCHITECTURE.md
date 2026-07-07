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
| `DataRegistry` | Maintains processed data tensors, module cache, load state, and `get_data(name)`. |

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
     -> validate returned shape [R, N]
     -> _apply_ops(item, loaded_window, lo_idx, hi_idx)
     -> write processed_cache

DataRegistry.get_data(name)
  -> returns processed_cache[resolved_name]
```

Supported built-in modules:

| Module | Responsibility |
|---|---|
| `builtin.factorsim` | Read factor or AshareCache data. 2-D sources enter the registry directly; 3-D sources enter the item ops pipeline as cube-like tensors after `nbar` bar-window selection. |
| `builtin.label` | Read DailyLabel data. |
| `builtin.alpha_parquet` | Read alpha parquet and align columns to `Universe.codes`. |
| `builtin.barra_style` | Read Barra CNE5 style exposure data. |

Supported item ops:

| Op | Responsibility |
|---|---|
| `cs_zscore` / `zscore` | Zscore on the requested axis. |
| `rank` | Rank on the requested axis; optional `pct` keeps percentile rank semantics. |
| `last` / `mean` / `std` / `sum` / `max` / `min` | Axis reducers. These remove the named axis and are how cube pipelines aggregate into final `[date, code]` panels. |
| `truncate` | Clamp by `min` and `max`. |
| `nan_to_num` / `fillna` | Fill NaN/inf with `value`. |
| `winsorize_by_quantile` | Quantile winsorization on the requested axis. |
| `normalize_by_max_abs` | Max-abs normalization on the requested axis. |
| `rolling_mean` / `rolling_std` | Rolling ops over an approved axis, defaulting to `date`. |
| `neut(name, ..., ratio)` | Neutralize against one or more data dependencies; optional final `ratio` defaults to `1.0` and may be a per-dependency ratio array. |

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
  -> feature cache lookup
  -> _build_feature(ds)
     -> build_raw_feature(ds)
        -> registry._ensure_range(factor_names, ds, ds)
        -> registry.get_data(name)[date_idx] for role="factor"
        -> stack factor [N, F_factor]
        -> cube_source.load_day(ds)
        -> concat cube [N, F]
        -> inf -> NaN
     -> preprocess_feature(feature, ds)
        -> cs_zscore across stocks per feature
        -> truncate(-4, 4)
        -> nan_to_num(0)
  -> write feature cache
  -> return [N, F]
```

Feature cache note:

- `feature cache` here means the final post-processed daily feature tensor keyed by `(ds, current_ti)`.
- It does not imply a persistent cache of raw 3-D minute cubes loaded from `builtin.factorsim`.

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
  -> process_feature_window(feature_window)
     -> _feature_available_mask(feature_window)
     -> nan_to_num(0)
     -> return feature_window[:, available_mask], available_mask
```

## ResearchLoader Hooks

`ResearchLoader` must inherit `ComboDataLoader`. Hooks are layered; overriding a parent organization function bypasses its child hooks unless the new implementation calls them.

```text
gen_feature(ds)
  -> build_raw_feature(ds)
  -> preprocess_feature(feature, ds)

gen_label(ds, ret_days)
  -> load label data
  -> aggregate ret_days returns
  -> build valid_mask
  -> preprocess_label(label_values, valid_mask, ds, ret_days)
```

Recommended override level:

| Goal | Hook |
|---|---|
| Replace zscore with rank, or change feature fill/truncate | `preprocess_feature(feature, ds)` |
| Change raw feature assembly, factor stack, cube concat, or raw derived features | `build_raw_feature(ds)` |
| Replace complete feature generation | `gen_feature(ds)` |
| Change label standardization, clipping, or sample weights | `preprocess_label(label_values, valid_mask, ds, ret_days)` |
| Replace complete label source/aggregation/valid-mask logic | `gen_label(ds, ret_days)` |
| Change prediction stock availability | `_feature_available_mask(feature_window)` |
| Change full prediction window processing | `process_feature_window(feature_window)` or `load_feature_window(end_ds, ts_days)` |

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
x: [ts_days, M, F]
y: [M]
w: [M]

Prediction input:
x_window: [ts_days, M, F]
available_mask: [N]
```
