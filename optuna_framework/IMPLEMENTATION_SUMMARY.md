# Optuna Framework Implementation Summary

## File List

- `optuna_framework/specs.py`: typed `RunPaths` for isolated inference runs.
- `optuna_framework/paths.py`: centralized repo-root and window-based run-path construction.
- `optuna_framework/config_renderer.py`: ElementTree XML patching, params/meta output, structured XML diff.
- `optuna_framework/runner.py`: `runCombo.py` subprocess runner and inference validation.
- `optuna_framework/metrics_parser.py`: full-window summary parsing and precise daily scoring-window metrics.
- `optuna_framework/aggregators.py`: single-window objective scoring, hard filters, baseline threshold builder, factor audit template.
- `optuna_framework/trial_meta.py`: atomic one-window `trial_meta.json` state updates.
- `optuna_framework/study_utils.py`: delayed Optuna/psutil/plotly helpers and report CSV writers.
- `optuna_framework/adapters/eg_torch_v1.py`: eg-torch search space and XML patching.
- `optuna_framework/studies/eg_torch_v1.py`: study constants, run windows, scoring window, fixed overrides.
- `optuna_framework/scripts/*.py`: dry-run render, baseline, smoke, Phase A/B/C, seed sanity, report export.
- `optuna_framework/tests/`: unit tests and fixtures.
- `optuna_framework/README.md`: manual workflow, dry-run, artifact, and window documentation.
- `optuna_framework/requirements-optuna.txt`: optional long-run dependencies.

## Key Design Decisions

- The old dated-run and study-spec abstractions were removed. Runs now carry
  explicit `run_start_ds`, `run_end_ds`, `score_start_ds`, and `score_end_ds`.
- Baseline performs one full-window run (`20200102-20240628`) and builds all
  thresholds from that run's `pnl_summary.csv` and `daily_pnl.csv`.
- Phase A/B run `20200102-20231229` and score only the precise daily slice
  `20210104-20231229`, keeping 2020 as rolling warmup.
- Phase C runs `20200102-20240628` and validates 2020, 2024H1, yearly tuning
  rows, and the full row from the same continuous inference pass.
- Optional dependencies are delayed: importing core modules does not require
  `optuna`, `psutil`, or Plotly.
- XML rendering starts from `eg-torch/config.xml` and patches fields with
  `xml.etree.ElementTree`; no string templates or `xmldiff`.
- Long scripts support `--dry-run`; dry-run prints commands and does not call
  `subprocess.run`.

## Directory Layout

- Baseline: `baseline/full_run/`
- Phase A: `trials/trial_<NNNNN>/`
- Phase B: `phase_b/candidate_<NN>/seed_<S>/`
- Phase C: `phase_c/<candidate_id>/seed_<S>/`

## Reports

- `reports/trials.csv`: Optuna trial state, objective, and params.
- `reports/scoring_metrics.csv`: one Phase A scoring-window metric row per trial.
- `reports/top10.csv`: best non-hard-filtered Phase A candidates.

## Verification

Run from the repository root with the local virtual environment:

```text
.\.venv\Scripts\python.exe -c "import optuna_framework"
.\.venv\Scripts\python.exe optuna_framework/scripts/dry_run_render.py
.\.venv\Scripts\python.exe optuna_framework/scripts/run_baseline.py --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_study.py --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_b.py --dry-run
.\.venv\Scripts\python.exe optuna_framework/scripts/run_phase_c.py --dry-run
.\.venv\Scripts\python.exe -m pytest optuna_framework/tests -q
rg -n "<legacy run abstraction patterns>" optuna_framework
```

The dry-run commands render or print planned paths only; they do not launch real
baseline or Phase A/B/C training jobs.

Latest local verification with `.\.venv\Scripts\python.exe`:

```text
import optuna_framework: OK
dry_run_render.py: OK
run_baseline.py --dry-run: OK
run_study.py --dry-run: OK
run_phase_b.py --dry-run: OK
run_phase_c.py --dry-run: OK
pytest optuna_framework/tests -q: 32 passed
legacy abstraction grep: no matches
```
