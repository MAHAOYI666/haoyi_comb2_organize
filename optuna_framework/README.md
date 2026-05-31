# Optuna Framework

This directory contains a generic Optuna hyperparameter-search scaffold. Model-specific search spaces live in model-owned plugins such as `eg-torch/optuna_plugin.py`; the framework renders isolated XML configs from the plugin's baseline config and does not modify that baseline config.

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
python3 optuna_framework/scripts/run_baseline.py --model-dir eg-torch --study-root "$STUDY_ROOT"
python3 optuna_framework/scripts/run_smoke.py --model-dir eg-torch --study-root "$STUDY_ROOT" --n-trials 3
python3 optuna_framework/scripts/run_study.py --model-dir eg-torch --study-root "$STUDY_ROOT" --n-trials 60
python3 optuna_framework/scripts/run_phase_b.py --model-dir eg-torch --study-root "$STUDY_ROOT"
python3 optuna_framework/scripts/run_phase_c.py --model-dir eg-torch --study-root "$STUDY_ROOT"
python3 optuna_framework/scripts/export_report.py --model-dir eg-torch --study-root "$STUDY_ROOT"
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

## Model-owned plugins

By default the scripts load `eg-torch/optuna_plugin.py`, but the canonical multi-model usage is to pass `--model-dir` or `--plugin`. To run another model, put an `optuna_plugin.py` next to that model's `config.xml` and `model.py`, then pass either:

```bash
python3 optuna_framework/scripts/run_study.py --model-dir eg-torch --dry-run
python3 optuna_framework/scripts/run_study.py --model-dir eg-lgbm --dry-run
python3 optuna_framework/scripts/run_study.py --plugin /root/autodl/0530.tcn/optuna_plugin.py --dry-run
python3 optuna_framework/scripts/run_study.py --plugin my-model/optuna_plugin.py --dry-run
```

A plugin must define `STUDY_SPEC` and either `create_adapter()` or `ADAPTER`. The adapter implements the generic `ModelAdapter` interface: `baseline_params()`, `suggest_params()`, `materialize_params()`, and `apply_params_to_xml()`. Phase B/C can optionally customize seed behavior by overriding `phase_seeds()`, `write_seeded_model()`, and `seeded_overrides()`. The `eg-torch` example overrides these hooks because torch training needs a seeded wrapper; `eg-lgbm` shows the simpler path where the default `combo.model.seed` override is enough; `/root/autodl/0530.tcn/optuna_plugin.py` is an external-plugin torch example that also needs the seeded-wrapper pattern.

Example layouts:

```text
my-model/
  config.xml
  model.py
  optuna_plugin.py
```

External flat repository example:

```text
/root/autodl/0530.tcn/
  config.xml
  Model.py
  optuna_plugin.py
```


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
.\.venv\Scripts\python.exe optuna_framework/scripts/run_baseline.py --model-dir eg-torch
```

3. Smoke study, manual long run:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/run_smoke.py --model-dir eg-torch
```

4. Phase A formal study:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/run_study.py --model-dir eg-torch --n-trials 60
```

5. Apply the seed patch documented at the top of
`scripts/seed_sanity_check.py`, then run:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/seed_sanity_check.py --model-dir eg-torch
```

6. Phase B and Phase C:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_b.py --model-dir eg-torch
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_c.py --model-dir eg-torch
```

7. Export report:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/export_report.py --model-dir eg-torch
```

## Dry-Run

Long-task scripts support `--dry-run`. Dry-run prints planned config paths and
`runCombo.py` commands without launching subprocesses. All scripts also support
`--model-dir` and `--plugin` for selecting a model-owned Optuna plugin:

```bash
.\.venv\Scripts\python.exe optuna_framework/scripts/run_baseline.py --model-dir eg-torch --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_smoke.py --model-dir eg-torch --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_study.py --model-dir eg-torch --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_b.py --model-dir eg-torch --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_c.py --model-dir eg-torch --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/export_report.py --model-dir eg-torch --dry-run
```

Use `--study-root` to override the default
`optuna_runs/study_eg_torch_v1` location. Each trial and segment receives a
separate config, output root, checkpoint root, log files, and snaptime.

## Design Notes

- Repo root discovery is centralized in `paths.get_repo_root()`.
- Model-specific Optuna definitions live in model-owned `optuna_plugin.py` files; `optuna_framework/plugin_loader.py` loads the selected plugin.
- XML rendering uses `xml.etree.ElementTree`; no string template or `xmldiff`
  dependency is required.
- `optuna`, `psutil`, and Plotly visualization imports are delayed until the
  script function that needs them.
- `baseline_thresholds.json` is generated only by manual baseline runs. Phase B
  and Phase C dry-runs fall back to test fixtures when the real file is absent.
- External torch plugins such as `/root/autodl/0530.tcn/optuna_plugin.py` may need a seeded wrapper for deterministic Phase B/C behavior if their base `Model.py` does not actually consume `seed` yet.
- When an external model's `load()` rebuilds structure from the current config and strict-loads weights, old manual checkpoints may be incompatible with new Optuna trial configs; prefer fresh study roots and isolated per-run checkpoint directories.
