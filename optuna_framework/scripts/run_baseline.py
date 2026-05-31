"""Manual baseline runner for all tuning, holdout, and full-period segments."""

from __future__ import annotations

import argparse

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.baseline_runner import run_baseline_evaluations
from optuna_framework.paths import resolve_study_root
from optuna_framework.scripts._script_common import add_study_args, load_study_and_adapter
from optuna_framework.scripts.run_study import _parse_devices


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run baseline evaluations for an Optuna plugin.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running runCombo.py")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of parallel baseline segments")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids or devices, for example: 0,1 or cuda:0,cuda:1")
    add_study_args(parser)
    return parser.parse_args()


def main() -> None:
    """Run or print baseline segment commands."""

    args = parse_args()
    study_spec, adapter, _plugin = load_study_and_adapter(args)
    study_root = resolve_study_root(args.study_root, study_spec.name)
    devices = _parse_devices(args.gpus)
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else (len(devices) if devices else 1)
    run_baseline_evaluations(study_spec, adapter, study_root, dry_run=args.dry_run, n_jobs=n_jobs, devices=devices)


if __name__ == "__main__":
    main()
