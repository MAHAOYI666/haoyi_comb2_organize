# Optuna Framework Implementation Summary

## File List

- `optuna_framework/specs.py`: typed `StudySpec`, `SegmentSpec`, and `RunPaths`
- `optuna_framework/paths.py`: centralized repo-root and run-path construction
- `optuna_framework/config_renderer.py`: ElementTree XML patching, params/meta output, structured XML diff
- `optuna_framework/runner.py`: `runCombo.py` subprocess runner and segment validation
- `optuna_framework/metrics_parser.py`: `pnl_summary.csv` parsing and full-period splitting
- `optuna_framework/aggregators.py`: objective scoring, hard filters, baseline threshold builder, factor audit template
- `optuna_framework/trial_meta.py`: atomic `trial_meta.json` state updates
- `optuna_framework/study_utils.py`: delayed Optuna/psutil/plotly helpers and report CSV writers
- `optuna_framework/adapters/eg_torch_v1.py`: 8-dimensional eg-torch search space and XML patching
- `optuna_framework/studies/eg_torch_v1.py`: segment and study specification
- `optuna_framework/scripts/*.py`: dry-run render, baseline, smoke, Phase A/B/C, seed sanity, report export
- `optuna_framework/tests/`: unit tests and mock fixtures
- `optuna_framework/README.md`: manual workflow and dry-run documentation
- `optuna_framework/requirements-optuna.txt`: optional long-run dependencies

## Key Design Decisions

- Optional dependencies are delayed: importing core modules does not require `optuna`, `psutil`, or Plotly.
- XML rendering starts from `eg-torch/config.xml` and patches fields with `xml.etree.ElementTree`; no string templates or `xmldiff`.
- Trial and segment path isolation is generated only through `paths.py`.
- Long scripts support `--dry-run`; dry-run prints commands and does not call `subprocess.run`.
- `dry_run_render.py` writes only under `/tmp/dry_run` and checks XML differences against a whitelist.
- `optuna_runs/` is ignored and was not created during engineering validation.

## Verification

`pytest optuna_framework/tests/`:

```text
18 passed in 0.44s
```

Import check:

```text
from optuna_framework.studies.eg_torch_v1 import STUDY_SPEC
study_eg_torch_v1
```

`dry_run_render.py` summary:

```text
rendered_config=D:\tmp\dry_run\baseline\seg01\config.xml
xml_diff_allowed=true
diff config.combo.runtime.@snaptime: 'example_torch' -> 'baseline_seg01'
diff config.constants.@checkpoint_root: 'checkpoints-torch' -> 'D:\\tmp\\dry_run\\baseline\\seg01\\checkpoints'
diff config.constants.@output_root: 'output-torch' -> 'D:\\tmp\\dry_run\\baseline\\seg01\\output'
diff config.strategy.@end_ds: 20240631 -> 20211231
diff config.strategy.@start_ds: 20200101 -> 20210104
```

Long-task script dry-runs were executed for baseline, smoke, Phase A, Phase B,
Phase C, seed sanity, and report export. They printed planned config paths and
commands only.

## Open Questions

- Real `baseline_thresholds.json` values must be produced by the manual baseline run before Phase A/B/C real execution.
- The seed patch is documented in `scripts/seed_sanity_check.py`; it has not been applied to `eg-torch/model.py`.
