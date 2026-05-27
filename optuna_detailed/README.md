# optuna_detailed

`optuna_detailed` is an isolated, config-driven Optuna workflow. It keeps the
original baseline / Phase A / Phase B / Phase C logic, while moving study setup
and search-space declarations into one XML file:

```text
optuna_detailed/config.xml
```

## Local Dry Run

Run from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe optuna_detailed/scripts/validate_config.py --config optuna_detailed/config.xml
.\.venv\Scripts\python.exe optuna_detailed/scripts/dry_run_render.py --config optuna_detailed/config.xml
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_baseline.py --config optuna_detailed/config.xml --dry-run
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_smoke.py --config optuna_detailed/config.xml --dry-run --n-trials 2
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_study.py --config optuna_detailed/config.xml --dry-run --n-trials 2
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_phase_b.py --config optuna_detailed/config.xml --dry-run
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_phase_c.py --config optuna_detailed/config.xml --dry-run
.\.venv\Scripts\python.exe optuna_detailed/scripts/export_report.py --config optuna_detailed/config.xml --dry-run
```

## Real Run Order

Real runs require the remote factor runtime:

```powershell
.\.venv\Scripts\python.exe optuna_detailed/scripts/validate_config.py --config optuna_detailed/config.xml
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_baseline.py --config optuna_detailed/config.xml
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_smoke.py --config optuna_detailed/config.xml --n-trials 10
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_study.py --config optuna_detailed/config.xml --n-trials 60
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_phase_b.py --config optuna_detailed/config.xml
.\.venv\Scripts\python.exe optuna_detailed/scripts/run_phase_c.py --config optuna_detailed/config.xml
.\.venv\Scripts\python.exe optuna_detailed/scripts/export_report.py --config optuna_detailed/config.xml
```

Real run scripts check `config_plan.txt` and refuse stale or blocked plans by
default. Use `--skip-plan-check` only for deliberate manual overrides.

## Config Notes

Tunable params must be declared in `<search_space>` and must exist in the
baseline XML under either `<combo><runtime ... />` or `<combo><model ... />`.
Virtual params use `target="none"` and must provide a `baseline` value.

The default config mirrors the existing `eg_torch_v1` setup, including the
same scoring objective, hard filter, Phase B seeds, and Phase C acceptance
rules.
