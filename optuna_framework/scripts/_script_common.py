"""Shared helpers for direct script execution from the repository root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def bootstrap_repo_imports() -> None:
    """Ensure ``optuna_framework`` is importable when scripts are run by path."""

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def add_study_args(parser: argparse.ArgumentParser) -> None:
    """Add generic model-owned plugin selection arguments."""

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-dir", default=None, help="Model directory containing optuna_plugin.py")
    group.add_argument("--plugin", default=None, help="Explicit optuna_plugin.py path")


def load_study_and_adapter(args: argparse.Namespace):
    """Load the selected Optuna study spec and adapter."""

    from optuna_framework.plugin_loader import load_plugin

    loaded = load_plugin(model_dir=getattr(args, "model_dir", None), plugin=getattr(args, "plugin", None))
    return loaded.study_spec, loaded.adapter, loaded


def fixture_thresholds_path() -> Path:
    """Return the test fixture thresholds path used by dry-run phase scripts."""

    from optuna_framework.paths import get_repo_root

    return get_repo_root() / "optuna_framework" / "tests" / "fixtures" / "baseline_thresholds.json"


def print_command(prefix: str, config_path: Path, cmd: list[str]) -> None:
    """Print a dry-run command in a stable format."""

    from optuna_framework.runner import command_to_string

    print(f"{prefix} config={config_path}")
    print(f"{prefix} cmd={command_to_string(cmd)}")
