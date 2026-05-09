# Optuna Framework

This directory contains an Optuna hyperparameter-search scaffold for the
`eg-torch` research model. It renders isolated XML configs from
`eg-torch/config.xml`; it does not modify the baseline config.

## Dependencies

Core modules avoid importing optional runtime packages at module import time.
Install long-run dependencies manually before running real studies:

```bash
.\.venv\Scripts\python.exe -m pip install -r optuna_framework/requirements-optuna.txt
```


## Remote Linux quick start

Run these commands from the repository root on the remote instance. Keep the same
`STUDY_ROOT` for every phase so the scripts can resume from the same SQLite DB
and reuse the generated reports.

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

If a segment fails, inspect the rendered config and subprocess logs under the
segment directory. For the first baseline segment those files are:

```bash
sed -n '1,220p' "$STUDY_ROOT/baseline/seg01/run.stderr.log"
sed -n '1,220p' "$STUDY_ROOT/baseline/seg01/run.stdout.log"
cat "$STUDY_ROOT/baseline/seg01/resolved_meta.json"
cat "$STUDY_ROOT/baseline/seg01/params.json"
```

The scripts also include the failed command, config path, log paths, and the
last log lines in `SegmentRunError` to make remote debugging easier. The
framework reads backtest summaries from `output/backtest/pnl_summary.csv` inside
each segment directory; `output/pnl_summary.csv` is not the expected location.

Optuna-rendered configs resolve `combo.paths.model_path` and `strategy.path` to
absolute paths before copying configs into isolated run directories. They also
disable the optional `alpha_analysis` step via
`combo.output.enable_alpha_analysis=false`. The objective only needs backtest
`pnl_summary.csv`, and skipping this IC dump avoids failures when a holdout
window has no valid label/mask coverage for IC analysis.

## Reading results and selecting parameters

Primary artifacts are written below `STUDY_ROOT`:

- `baseline/baseline_thresholds.json`: baseline metrics plus hard-filter limits.
- `reports/trials.csv`: one row per Optuna trial, with `value` as the objective
  to maximize and `param_*` columns as the searched parameters.
- `reports/segment_metrics.csv`: per-trial, per-segment sharpe/drawdown/return
  diagnostics from `trial_meta.json`.
- `reports/top10.csv`: best non-hard-filtered Phase A candidates.
- `reports/param_importance.html`, `reports/parallel_coordinate.html`, and
  `reports/optimization_history.html`: optional Optuna/Plotly visualizations.
- `phase_b/results.json` and `phase_b/survivors.json`: three-seed tuning-year
  stability checks for selected candidates.
- `phase_c/results.json`: full-period and holdout validation for Phase B
  survivors.
- `REPORT.md`: final human-readable summary and recommendation.

Use this quick CLI view after Phase A:

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path
root = Path(__import__('os').environ['STUDY_ROOT'])
print(pd.read_csv(root / 'reports' / 'top10.csv').head(10).to_string(index=False))
print(pd.read_csv(root / 'reports' / 'segment_metrics.csv').head(30).to_string(index=False))
PY
```

Choose final parameters only after Phase C: prefer the first `accepted=true`
entry in `phase_c/results.json` or the recommendation in `REPORT.md`. If Phase
C has no accepted candidate, keep the baseline parameters or manually review the
Phase B/C rejection reasons before changing production config.

## Manual Run Order

Run from the repository root.

1. Render-only sanity check:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/dry_run_render.py
```

Expected output shape:

```text
rendered_config=D:\tmp\dry_run\baseline\seg01\config.xml
xml_diff_allowed=true
diff config.combo.output.@enable_alpha_analysis: None -> False
diff config.combo.paths.@model_path: 'model.py' -> '.../eg-torch/model.py'
diff config.combo.runtime.@snaptime: 'example_torch' -> 'baseline_seg01'
diff config.constants.@checkpoint_root: 'checkpoints-torch' -> '...\\checkpoints'
diff config.constants.@output_root: 'output-torch' -> '...\\output'
diff config.strategy.@end_ds: 20240631 -> 20211231
diff config.strategy.@path: '../alpha_strategy.py' -> '.../alpha_strategy.py'
diff config.strategy.@start_ds: 20200101 -> 20210104
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

5. Apply the seed patch documented at the top of
`scripts/seed_sanity_check.py`, then run:

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

Use `--study-root` to override the default
`optuna_runs/study_eg_torch_v1` location. Each trial and segment receives a
separate config, output root, checkpoint root, log files, and snaptime.

## Design Notes

- Repo root discovery is centralized in `paths.get_repo_root()`.
- XML rendering uses `xml.etree.ElementTree`; no string template or `xmldiff`
  dependency is required.
- `optuna`, `psutil`, and Plotly visualization imports are delayed until the
  script function that needs them.
- `baseline_thresholds.json` is generated only by manual baseline runs. Phase B
  and Phase C dry-runs fall back to test fixtures when the real file is absent.
- `ensure_factor_audit_template(study_root)` creates the audit template only
  when baseline is run for real.
