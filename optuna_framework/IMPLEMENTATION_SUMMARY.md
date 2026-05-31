# Optuna Framework Implementation Summary

## File List

- `optuna_framework/specs.py`: typed `StudySpec`, `SegmentSpec`, and `RunPaths`
- `optuna_framework/paths.py`: centralized repo-root and run-path construction
- `optuna_framework/plugin_loader.py`: model-owned plugin discovery and loading
- `optuna_framework/config_renderer.py`: ElementTree XML patching, params/meta output, structured XML diff
- `optuna_framework/runner.py`: `runCombo.py` subprocess runner and segment validation
- `optuna_framework/metrics_parser.py`: `pnl_summary.csv` parsing and full-period splitting
- `optuna_framework/aggregators.py`: objective scoring, hard filters, baseline threshold builder
- `optuna_framework/trial_meta.py`: atomic `trial_meta.json` state updates
- `optuna_framework/study_utils.py`: delayed Optuna/psutil/plotly helpers and report CSV writers
- `eg-torch/optuna_plugin.py`: canonical torch example plugin, including seeded wrapper logic for Phase B/C
- `eg-lgbm/optuna_plugin.py`: canonical LightGBM example plugin using default Phase B/C seed hooks
- `optuna_framework/scripts/*.py`: dry-run render, baseline, smoke, Phase A/B/C, seed sanity, report export
- `optuna_framework/tests/`: unit tests and mock fixtures
- `optuna_framework/README.md`: plugin-first workflow and dry-run documentation
- `optuna_framework/requirements-optuna.txt`: optional long-run dependencies

## Key Design Decisions

- Optional dependencies are delayed: importing core modules does not require `optuna`, `psutil`, or Plotly.
- Model-specific Optuna behavior lives in model-owned `optuna_plugin.py` files loaded through `optuna_framework/plugin_loader.py`.
- XML rendering starts from the selected plugin's `STUDY_SPEC.baseline_config_path` and patches fields with `xml.etree.ElementTree`; no string templates or `xmldiff`.
- Trial and segment path isolation is generated only through `paths.py`.
- Long scripts support `--dry-run`; dry-run prints commands and does not call `subprocess.run`.
- `eg-torch` needs a seeded model wrapper for deterministic Phase B/C checks; `eg-lgbm` can reuse the default `combo.model.seed` override because its model already consumes `seed` directly.
- `dry_run_render.py` writes only under `/tmp/dry_run` and checks XML differences against a whitelist.
- `optuna_runs/` is ignored and was not created during engineering validation.

## Verification

Current verification target:

```text
pytest optuna_framework/tests/
```

Canonical loader checks should use:

```text
load_plugin(model_dir="eg-torch")
load_plugin(model_dir="eg-lgbm")
```

Example dry-run checks:

```text
python optuna_framework/scripts/run_study.py --model-dir eg-torch --dry-run --n-trials 2
python optuna_framework/scripts/run_study.py --model-dir eg-lgbm --dry-run --n-trials 2
python optuna_framework/scripts/run_phase_b.py --model-dir eg-lgbm --dry-run
python optuna_framework/scripts/run_phase_c.py --model-dir eg-lgbm --dry-run
```

## Open Questions

- Real `baseline_thresholds.json` values must still be produced by a manual baseline run before Phase A/B/C real execution.
- The legacy wrapper modules remain temporarily for compatibility; once downstream imports are cleaned up, they can be removed in a later pass.
