"""Manual baseline runner for all tuning, holdout, and full-period segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.aggregators import build_baseline_thresholds, ensure_factor_audit_template
from optuna_framework.config_renderer import render_config
from optuna_framework.metrics_parser import parse_full_period, parse_segment_metrics
from optuna_framework.paths import build_baseline_run_paths, resolve_study_root
from optuna_framework.runner import build_run_command, run_segment
from optuna_framework.scripts._script_common import adapter_for_name, print_command
from optuna_framework.studies.eg_torch_v1 import STUDY_SPEC


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run baseline evaluations for eg_torch_v1.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running runCombo.py")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    return parser.parse_args()


def main() -> None:
    """Run or print baseline segment commands."""

    args = parse_args()
    study_root = resolve_study_root(args.study_root, STUDY_SPEC.name)
    adapter = adapter_for_name(STUDY_SPEC.adapter_name)
    params = adapter.baseline_params()

    if args.dry_run:
        print(f"[DRY-RUN] baseline study_root={study_root}")
        for segment in STUDY_SPEC.baseline_segments():
            run_paths = build_baseline_run_paths(study_root, segment)
            print_command(f"[DRY-RUN] baseline/{segment.name}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    ensure_factor_audit_template(study_root)
    by_segment = {}
    full_metrics = None
    full_by_year = {}
    for segment in STUDY_SPEC.baseline_segments():
        run_paths = build_baseline_run_paths(study_root, segment)
        render_config(STUDY_SPEC.baseline_config_path, run_paths, adapter, params, STUDY_SPEC.fixed_overrides)
        metrics = run_segment(run_paths)
        if segment.name == "full_period":
            full_by_year = parse_full_period(run_paths.pnl_summary_path)
            full_metrics = full_by_year["full"]
        else:
            by_segment[segment.name] = metrics

    if full_metrics is None:
        raise RuntimeError("full_period baseline metrics were not produced")
    thresholds = build_baseline_thresholds(by_segment, full_metrics, full_by_year)
    threshold_path = study_root / "baseline" / "baseline_thresholds.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"baseline_thresholds={threshold_path}")


if __name__ == "__main__":
    main()

