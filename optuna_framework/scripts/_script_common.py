"""Shared helpers for direct script execution from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def bootstrap_repo_imports() -> None:
    """Ensure ``optuna_framework`` is importable when scripts are run by path."""

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def adapter_for_name(name: str) -> Any:
    """Return an adapter instance by registered name."""

    from optuna_framework.adapters.eg_torch_v1 import EgTorchV1Adapter

    if name == EgTorchV1Adapter.name:
        return EgTorchV1Adapter()
    raise KeyError(f"unknown adapter: {name}")


def fixture_thresholds_path() -> Path:
    """Return the test fixture thresholds path used by dry-run phase scripts."""

    from optuna_framework.paths import get_repo_root

    return get_repo_root() / "optuna_framework" / "tests" / "fixtures" / "baseline_thresholds.json"


def print_command(prefix: str, config_path: Path, cmd: list[str]) -> None:
    """Print a dry-run command in a stable format."""

    from optuna_framework.runner import command_to_string

    print(f"{prefix} config={config_path}")
    print(f"{prefix} cmd={command_to_string(cmd)}")

