# Optuna Framework

This directory contains an Optuna hyperparameter-search scaffold for the
`eg-torch` research model. It renders isolated XML configs from
`eg-torch/config.xml`; it does not modify the baseline config.

## Window Design

The framework uses one rolling training setup. Training data still starts from
`combo.loader.data_start_ds=20160101`; each phase only changes the rendered
inference window and the metrics window used for scoring.

- Baseline: run `20200102-20240628`, then cut all thresholds from that one run.
- Phase A: run `20200102-20231229`, score the precise daily slice
  `20210104-20231229`.
- Phase B: same run and scoring windows as Phase A, repeated for seeds
  `42/43/44`.
- Phase C: run `20200102-20240628`, validate yearly rows plus the full row.

The old dated-run abstraction has been removed. Runtime paths are represented
by `RunPaths` with explicit `run_window` and `score_window` fields.

## Dependencies

Core modules avoid importing optional runtime packages at module import time.
Install long-run dependencies manually before running real studies:

```bash
.\.venv\Scripts\python.exe -m pip install -r optuna_framework/requirements-optuna.txt
```

## Manual Run Order

Run from the repository root. The commands below use the local Windows virtual
environment expected by this repository.

1. Render-only sanity check:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/dry_run_render.py
```

2. Baseline evaluation, manual long run:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/run_baseline.py
```

3. Smoke study, manual long run:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/run_smoke.py
```

4. Phase A formal study:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/run_study.py --n-trials 60
```

5. Same-seed sanity check:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/seed_sanity_check.py
```

6. Phase B and Phase C:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_b.py
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_c.py
```

7. Export report:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/export_report.py
```

## Dry-Run

Long-task scripts support `--dry-run`. Dry-run prints planned config paths and
`runCombo.py` commands without launching subprocesses:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/run_baseline.py --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_smoke.py --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_study.py --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_b.py --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_c.py --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/export_report.py --dry-run
```

Expected `dry_run_render.py` output shape:

```text
rendered_config=...\optuna_runs\dry_run_render\baseline\full_run\config.xml
run_window=20200102-20240628
scoring_window=20210104-20231229
xml_diff_allowed=true
diff config.combo.output.@enable_alpha_analysis: None -> False
diff config.combo.paths.@model_path: 'model.py' -> '...\eg-torch\model.py'
diff config.combo.runtime.@snaptime: 'example_torch' -> 'baseline_full_run'
diff config.constants.@checkpoint_root: 'checkpoints-torch' -> '...\baseline\full_run\checkpoints'
diff config.constants.@output_root: 'output-torch' -> '...\baseline\full_run\output'
diff config.strategy.@end_ds: 20240631 -> 20240628
diff config.strategy.@path: '../alpha_strategy.py' -> '...\alpha_strategy.py'
diff config.strategy.@start_ds: 20200101 -> 20200102
```

## Directory Layout

Use `--study-root` to override the default
`optuna_runs/study_eg_torch_v1` location. Each run receives a separate config,
output root, checkpoint root, log files, and snaptime.

- Baseline: `baseline/full_run/`
- Phase A: `trials/trial_<NNNNN>/`
- Phase B: `phase_b/candidate_<NN>/seed_<S>/`
- Phase C: `phase_c/<candidate_id>/seed_<S>/`

If a run fails, inspect the rendered config and subprocess logs under the run
directory. For the baseline run these files are:

```bash
Get-Content "$env:STUDY_ROOT\baseline\full_run\run.stderr.log" -TotalCount 220
Get-Content "$env:STUDY_ROOT\baseline\full_run\run.stdout.log" -TotalCount 220
Get-Content "$env:STUDY_ROOT\baseline\full_run\resolved_meta.json"
Get-Content "$env:STUDY_ROOT\baseline\full_run\params.json"
```

The scripts include the failed command, config path, log paths, and the last log
lines in `InferenceRunError`. The framework reads backtest summaries from
`output/backtest/pnl_summary.csv` inside each run directory; `output/pnl_summary.csv`
is not the expected location.

## Artifacts

Primary artifacts are written below `STUDY_ROOT`:

- `baseline/baseline_thresholds.json`: tuning-period metrics, holdout slices,
  full-period metrics, yearly full-period metrics, and hard-filter limits.
- `reports/trials.csv`: one row per Optuna trial, with `value` as the objective
  to maximize and `param_*` columns as the searched parameters.
- `reports/scoring_metrics.csv`: one Phase A scoring-window row per trial.
- `reports/top10.csv`: best non-hard-filtered Phase A candidates.
- `reports/param_importance.html`, `reports/parallel_coordinate.html`, and
  `reports/optimization_history.html`: optional Optuna/Plotly visualizations.
- `phase_b/results.json` and `phase_b/survivors.json`: three-seed scoring-window
  stability checks for selected candidates.
- `phase_c/results.json`: full-period and holdout validation for Phase B
  survivors.
- `REPORT.md`: final human-readable summary and recommendation.

Use this quick CLI view after Phase A:

```bash
.\.venv\Scripts\python.exe -c "import os, pandas as pd; from pathlib import Path; root=Path(os.environ['STUDY_ROOT']); print(pd.read_csv(root/'reports'/'top10.csv').head(10).to_string(index=False)); print(pd.read_csv(root/'reports'/'scoring_metrics.csv').head(30).to_string(index=False))"
```

Choose final parameters only after Phase C: prefer the first `accepted=true`
entry in `phase_c/results.json` or the recommendation in `REPORT.md`. If Phase C
has no accepted candidate, keep the baseline parameters or manually review the
Phase B/C rejection reasons before changing production config.

## Baseline Thresholds

`baseline_thresholds.json` is generated from one full-window baseline run:

- `tuning_period`: daily slice `20210104-20231229` from `daily_pnl.csv`.
- `by_segment.holdout_2020`: the 2020 row from full-window `pnl_summary.csv`.
- `by_segment.holdout_2024h1`: the 2024H1 row from full-window `pnl_summary.csv`.
- `full_period`: the full row plus `by_year` rows from full-window
  `pnl_summary.csv`.
- `hard_filter.min_sharpe_threshold`: `tuning_period.sharpe_idx - 0.3`.
- `hard_filter.max_dd_threshold`: `tuning_period.dd_li * 1.3`.

Phase A objective is the single scoring-window `sharpe_idx` unless the hard
filter fires, in which case the objective is `-10.0`.

## Remote Linux Quick Start

On a remote Linux instance, use the platform Python selected for that machine
and keep the same `STUDY_ROOT` for every phase:

```bash
cd /root/autodl-tmp/haoyi_comb2_organize
python3 -m pip install -r optuna_framework/requirements-optuna.txt
export STUDY_ROOT=/root/autodl-tmp/haoyi_comb2_organize/optuna_runs/study_eg_torch_v1
python3 optuna_framework/scripts/dry_run_render.py
python3 optuna_framework/scripts/run_baseline.py --study-root "$STUDY_ROOT"
python3 optuna_framework/scripts/run_smoke.py --study-root "$STUDY_ROOT" --n-trials 3
python3 optuna_framework/scripts/run_study.py --study-root "$STUDY_ROOT" --n-trials 60
python3 optuna_framework/scripts/run_phase_b.py --study-root "$STUDY_ROOT"
python3 optuna_framework/scripts/run_phase_c.py --study-root "$STUDY_ROOT"
python3 optuna_framework/scripts/export_report.py --study-root "$STUDY_ROOT"
```
