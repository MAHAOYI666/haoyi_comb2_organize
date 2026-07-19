"""Manual baseline runner for a detailed Optuna study."""

from __future__ import annotations

import argparse
import json

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.aggregators import build_baseline_thresholds, ensure_factor_audit_template
from optuna_framework.config_renderer import render_config
from optuna_framework.metrics_parser import parse_full_period, parse_window_metrics
from optuna_framework.paths import build_baseline_run_paths
from optuna_framework.runner import build_run_command, run_inference
from optuna_framework.scripts._script_common import add_common_config_args, add_plan_check_arg, ensure_plan_for_args, load_config_from_args, print_command
from optuna_framework.search_space import ConfigDrivenAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full-window baseline evaluation.")
    add_common_config_args(parser)
    add_plan_check_arg(parser)
    parser.add_argument("--dry-run", action="store_true", help="Print planned command without running runCombo.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_args(args)
    ensure_plan_for_args(config, args, dry_run=args.dry_run)
    adapter = ConfigDrivenAdapter(config)
    params = adapter.baseline_params()
    run_paths = build_baseline_run_paths(config.study_root, config.full_run_window, config.scoring_window)

    if args.dry_run:
        print(f"[DRY-RUN] baseline study_root={config.study_root}")
        print(f"[DRY-RUN] run_window={config.full_run_window[0]}-{config.full_run_window[1]}")
        print(f"[DRY-RUN] scoring_window={config.scoring_window[0]}-{config.scoring_window[1]}")
        print_command("[DRY-RUN] baseline/full_run", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    ensure_factor_audit_template(config.study_root)
    render_config(config, run_paths, adapter, params)
    run_inference(run_paths)
    full_by_year = parse_full_period(run_paths.pnl_summary_path)
    tuning_period = parse_window_metrics(
        run_paths.pnl_summary_path,
        run_start_ds=config.full_run_window[0],
        run_end_ds=config.full_run_window[1],
        score_start_ds=config.scoring_window[0],
        score_end_ds=config.scoring_window[1],
    )
    thresholds = build_baseline_thresholds(full_by_year, tuning_period)
    threshold_path = config.study_root / "baseline" / "baseline_thresholds.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"baseline_thresholds={threshold_path}")


if __name__ == "__main__":
    main()
