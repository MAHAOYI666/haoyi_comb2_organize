"""Manual baseline runner for the single full-window inference run."""

from __future__ import annotations

import argparse
import json

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.aggregators import build_baseline_thresholds, ensure_factor_audit_template
from optuna_framework.config_renderer import render_config
from optuna_framework.metrics_parser import parse_full_period, parse_window_metrics
from optuna_framework.paths import build_baseline_run_paths, resolve_study_root
from optuna_framework.runner import build_run_command, run_inference
from optuna_framework.scripts._script_common import adapter_for_name, print_command
from optuna_framework.studies.eg_torch_v1 import (
    ADAPTER_NAME,
    BASELINE_CONFIG_PATH,
    FIXED_OVERRIDES,
    FULL_RUN_WINDOW,
    SCORING_WINDOW,
    STUDY_NAME,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the full-window baseline evaluation for eg_torch_v1.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned command without running runCombo.py")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    return parser.parse_args()


def main() -> None:
    """Run or print the baseline command."""

    args = parse_args()
    study_root = resolve_study_root(args.study_root, STUDY_NAME)
    adapter = adapter_for_name(ADAPTER_NAME)
    params = adapter.baseline_params()
    run_paths = build_baseline_run_paths(study_root, FULL_RUN_WINDOW, SCORING_WINDOW)

    if args.dry_run:
        print(f"[DRY-RUN] baseline study_root={study_root}")
        print(f"[DRY-RUN] run_window={FULL_RUN_WINDOW[0]}-{FULL_RUN_WINDOW[1]}")
        print(f"[DRY-RUN] scoring_window={SCORING_WINDOW[0]}-{SCORING_WINDOW[1]}")
        print_command("[DRY-RUN] baseline/full_run", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    ensure_factor_audit_template(study_root)
    render_config(BASELINE_CONFIG_PATH, run_paths, adapter, params, FIXED_OVERRIDES)
    run_inference(run_paths)
    full_by_year = parse_full_period(run_paths.pnl_summary_path)
    tuning_period = parse_window_metrics(
        run_paths.pnl_summary_path,
        run_start_ds=FULL_RUN_WINDOW[0],
        run_end_ds=FULL_RUN_WINDOW[1],
        score_start_ds=SCORING_WINDOW[0],
        score_end_ds=SCORING_WINDOW[1],
    )
    thresholds = build_baseline_thresholds(full_by_year, tuning_period)
    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"baseline_thresholds={threshold_path}")


if __name__ == "__main__":
    main()
