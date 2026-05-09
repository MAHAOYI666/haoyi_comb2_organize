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
diff config.combo.runtime.@snaptime: 'example_torch' -> 'baseline_seg01'
diff config.constants.@checkpoint_root: 'checkpoints-torch' -> '...\\checkpoints'
diff config.constants.@output_root: 'output-torch' -> '...\\output'
diff config.strategy.@end_ds: 20240631 -> 20211231
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
