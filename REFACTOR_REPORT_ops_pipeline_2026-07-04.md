# Ops Pipeline Refactor Report

Date: 2026-07-04

## Summary

This refactor removed the old dedicated `reducer` concept from the 3D `builtin.factorsim` path and moved the data flow to an ops-only pipeline:

```text
raw 2D source -> ops -> [date, code]
raw 3D cube -> nbar pre-window -> ops -> [date, code]
```

The downstream contract did not change:

- `DataRegistry.processed_cache[name]` still stores final `[date, code]`
- `DataLoader` and downstream feature assembly still consume `[date, code]`

The main goal was to make cube aggregation and transformation use the same operator surface instead of a separate `reducer` mechanism.

## Scope

In scope:

- `vendor/comb2/src/DataRegistry.py`
- `vendor/comb2/src/op_utils.py`
- `vendor/comb2/tests/test_codec.py`
- `tests/test_comb2_simbase.py`
- `config.human`
- `README.md`
- `vendor/comb2/docs/ARCHITECTURE.md`

Out of scope:

- `DataLoader` feature/label contract changes
- runtime/backtest/model pipeline changes
- compatibility layer for old `reducer` syntax

## Public API Changes

### 1. `reducer` removed from the new interface

Old style:

```xml
<item ... nbar="5" reducer="mean" />
```

New style:

```xml
<item ... nbar="5">
  <op name="mean" axis="bar" />
</item>
```

### 2. `nbar` kept as pre-op window selection

`nbar` now has only one meaning:

- for 3D cube items, before ops run, keep the last `nbar` positions on the `bar` axis
- `nbar` does not aggregate or reduce any axis

### 3. `freq` currently only provides default `nbar`

For 3D items:

- explicit `nbar` is allowed
- if `nbar` is omitted, `freq` can provide the default full bar count

Current hardcoded support:

- `1m` / `1min` -> `239`
- `5m` / `5min` -> `49`

Important:

- `freq` is not yet the loader-path selector
- 2D vs 3D load path is still determined by source dimensionality (`reader.n_levels`)

### 4. Named axes are now part of the interface

Supported semantic axes:

- `date`
- `bar`
- `code`

Integer axes still work in some places for compatibility inside the code path, but the intended interface is semantic-axis-first.

### 5. Old ops removed/replaced

Removed from the item-op path:

- `delay`
- `ts_mean`
- `ts_avg`

Replacement:

- `rolling_mean`
- `rolling_std`

### 6. Final pipeline contract is enforced

For every item, the ops pipeline must end as:

```text
[date, code]
```

If the final simulated axes are not `[date, code]`, the registry raises a config/runtime error before caching the item.

## Internal Refactor Details

### A. Lightweight axis simulator added

File:

- `vendor/comb2/src/DataRegistry.py`

Added:

- `TensorSpec`
- `_simulate_ops(...)`

Purpose:

- simulate axis evolution before execution
- distinguish `date`, `bar`, `code`
- reject illegal pipelines early

This is intentionally lightweight and not a full graph/IR system.

### B. Raw data tensor handling generalized

Added:

- `_as_data_tensor(...)`

Behavior:

- accepts 2D or 3D tensors from data modules
- final cached result is still required to be 2D

### C. 3D factorsim load path changed

Old path:

- `load_reduced(...)`
- `_reduce_window(...)`
- explicit `reducer`

New path:

- `load_cube(...)`
- no dedicated reducer stage
- cube is loaded as `[date, bar, code]`
- `nbar` applied before ops
- reduction happens through normal ops like `mean(axis="bar")`

### D. Operator families clarified

The code now effectively separates these categories:

#### Shape-preserving axis-aware ops

- `cs_zscore` / `zscore`
- `rank`
- `truncate`
- `nan_to_num` / `fillna`
- `winsorize_by_quantile`
- `normalize_by_max_abs`

#### Reduction ops

- `last`
- `mean`
- `std`
- `sum`
- `max`
- `min`

These remove an axis.

#### Rolling ops

- `rolling_mean`
- `rolling_std`

#### Dependency op

- `neut`

`neut` keeps fixed semantics on the `code` axis and does not expose axis as a public parameter.

### E. `op_utils.py` extended

Added/reworked:

- pure reduction helpers: `reduce_last`, `reduce_mean`, `reduce_std`, `reduce_sum`, `reduce_max`, `reduce_min`
- rolling helpers: `rolling_mean`, `rolling_std`
- `rank` implementation no longer depends on pandas ranking for the main path

## Deleted / Obsoleted Logic

Deleted from the runtime path:

- dedicated `reducer` dispatch
- `_reduce_window(...)`
- old `delay`
- old `ts_mean` / `ts_avg`

The new runtime no longer relies on a separate reducer abstraction.

## Behavior Notes

### 1. 2D item reductions on `code` are illegal unless the pipeline restores `[date, code]`

Example:

```xml
<op name="mean" axis="code" />
```

on a 2D item ends as `[date]`, so it is rejected.

This is intentional under the current final-contract rule.

### 2. 3D cube item example

Valid:

```xml
<item ... freq="1m">
  <op name="mean" axis="bar" />
</item>
```

Invalid:

```xml
<item ... freq="1m">
  <op name="cs_zscore" axis="code" />
</item>
```

because the pipeline would still end as `[date, bar, code]`.

### 3. `freq` is not yet a routing key

Current status:

- source dimensionality still decides whether we load as 2D or 3D
- `freq` only provides default full-bar `nbar`

This was kept intentionally to avoid adding a second major routing abstraction in the same refactor.

## Files Changed

Core:

- `vendor/comb2/src/DataRegistry.py`
- `vendor/comb2/src/op_utils.py`

Tests:

- `vendor/comb2/tests/test_codec.py`
- `tests/test_comb2_simbase.py`

Docs:

- `config.human`
- `README.md`
- `vendor/comb2/docs/ARCHITECTURE.md`

## Test Coverage Added/Updated

Updated tests cover:

- rolling op semantics
- axis-aware ops on 2D panels
- invalid pipelines that do not end as `[date, code]`
- config -> LoaderConfig -> ComboDataLoader -> 3D `builtin.factorsim` end-to-end path
- 3D cube path with explicit `nbar`
- 3D cube path with `freq`-provided default `nbar`
- failure when 3D source has neither `nbar` nor `freq`
- current `ti` sensitivity for 3D cube windows
- 3D source-time-axis selection and source-column reindexing
- date-axis rolling raw lookback reload after a processed-cache hit
- `builtin.factor` is accepted at config parse time and fails later during registry module resolution

## Verification Run

Executed:

```bash
python -m pytest tests/test_cli_entrypoints.py tests/test_comb2_simbase.py vendor/comb2/tests/test_codec.py
```

Result:

- `97 passed`
- `1 skipped`

Also executed:

```bash
python -m py_compile comboHelloWorld.py config.py vendor/comb2/src/DataRegistry.py vendor/comb2/src/op_utils.py vendor/comb2/src/DataLoader.py vendor/comb2/src/ComboBase.py
git diff --check
```

Coverage command:

```bash
python -m coverage erase && \
python -m coverage run --source=config,comboHelloWorld,runCombo,runEval,vendor/comb2/src \
  -m pytest tests/test_cli_entrypoints.py tests/test_comb2_simbase.py vendor/comb2/tests/test_codec.py && \
python -m coverage report -m config.py comboHelloWorld.py runCombo.py runEval.py \
  vendor/comb2/src/DataRegistry.py vendor/comb2/src/op_utils.py \
  vendor/comb2/src/DataLoader.py vendor/comb2/src/ComboBase.py
```

Coverage result:

```text
config.py                         90%
vendor/comb2/src/DataRegistry.py  76%
vendor/comb2/src/op_utils.py      66%
vendor/comb2/src/DataLoader.py    60%
vendor/comb2/src/ComboBase.py     29%
runCombo.py                       26%
comboHelloWorld.py                 0%
runEval.py                         0%
TOTAL                             52%
```

Note: `comboHelloWorld.py` and `runEval.py` are exercised by CLI tests through subprocesses, but this coverage run does not collect subprocess coverage.

## Review Checklist

Reviewers should check:

- whether the final `[date, code]` contract is the right strictness level
- whether 2D item reductions should stay illegal unless re-expanded
- whether `freq` should remain only a default-`nbar` provider or later become the loader selector
- whether the hardcoded `FREQ_BAR_COUNTS` mapping should move to a more explicit metadata source
- whether additional window-selection ops such as `slice/head/tail` are needed next

## Known Remaining Boundary

The main intentional non-change is:

- `freq` does not yet select the loader path (`day` vs `1m` vs `5m`)

If that is needed later, it should be treated as a separate interface upgrade rather than folded into this refactor silently.
