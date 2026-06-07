"""Shared helpers for direct detailed Optuna script execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def bootstrap_repo_imports() -> None:
    """Ensure ``optuna_framework`` is importable when scripts are run by path."""

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    """Add config and study-root arguments shared by detailed scripts."""

    parser.add_argument("--config", default=None, help="Path to optuna_framework XML config")
    parser.add_argument("--study-root", default=None, help="Override configured/default study root")


def load_config_from_args(args: argparse.Namespace):
    """Load StudyConfig from common CLI args."""

    from optuna_framework.study_config import load_study_config

    return load_study_config(args.config, study_root_override=args.study_root)


def ensure_plan_for_args(config, args: argparse.Namespace, dry_run: bool = False):
    """Check or generate the confirmation plan according to run mode."""

    from optuna_framework.config_validator import assert_plan_ok, ensure_config_plan, write_config_plan

    if dry_run:
        plan = ensure_config_plan(config, write=True)
        print(f"[DRY-RUN] config_plan={write_config_plan(plan)}")
        return plan
    return assert_plan_ok(config, skip_plan_check=getattr(args, "skip_plan_check", False))


def add_plan_check_arg(parser: argparse.ArgumentParser) -> None:
    """Add the explicit plan-check bypass argument for real runs."""

    parser.add_argument("--skip-plan-check", action="store_true", help="Bypass saved plan_hash check for real runs")


def fixture_thresholds_path() -> Path:
    """Return the test fixture thresholds path used by dry-run phase scripts."""

    from optuna_framework.paths import get_repo_root

    return get_repo_root() / "optuna_framework" / "tests" / "fixtures" / "baseline_thresholds.json"


def print_command(prefix: str, config_path: Path, cmd: list[str]) -> None:
    """Print a dry-run command in a stable format."""

    from optuna_framework.runner import command_to_string

    print(f"{prefix} config={config_path}")
    print(f"{prefix} cmd={command_to_string(cmd)}")
