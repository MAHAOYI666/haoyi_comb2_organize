"""Run rendered ``runCombo.py`` configs and validate resulting metrics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from optuna_framework.metrics_parser import parse_segment_metrics
from optuna_framework.paths import get_repo_root
from optuna_framework.specs import RunPaths
from optuna_framework.trial_meta import update_segment


class SegmentRunError(RuntimeError):
    """Raised when one segment subprocess fails or emits invalid metrics."""


def build_run_command(config_path: str | Path) -> list[str]:
    """Build the canonical runCombo command."""

    repo_root = get_repo_root()
    return [
        sys.executable,
        str(repo_root / "runCombo.py"),
        str(Path(config_path).expanduser().resolve()),
    ]


def command_to_string(cmd: list[str]) -> str:
    """Return a display string for a subprocess command."""

    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def run_segment(run_paths: RunPaths) -> object:
    """Execute one rendered segment and return parsed metrics."""

    cmd = build_run_command(run_paths.config_path)
    run_paths.segment_dir.mkdir(parents=True, exist_ok=True)
    with run_paths.stdout_path.open("w", encoding="utf-8") as stdout_file, run_paths.stderr_path.open("w", encoding="utf-8") as stderr_file:
        proc = subprocess.run(
            cmd,
            cwd=str(get_repo_root()),
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )

    try:
        if proc.returncode != 0:
            raise SegmentRunError(f"runCombo failed for {run_paths.segment.name} with returncode={proc.returncode}")
        if not run_paths.pnl_summary_path.exists() or run_paths.pnl_summary_path.stat().st_size <= 0:
            raise SegmentRunError(f"missing or empty pnl_summary.csv: {run_paths.pnl_summary_path}")
        metrics = parse_segment_metrics(
            run_paths.pnl_summary_path,
            start_ds=run_paths.segment.start_ds,
            end_ds=run_paths.segment.end_ds,
            role=run_paths.segment.role,
        )
    except Exception as exc:
        _mark_failed(run_paths)
        if isinstance(exc, SegmentRunError):
            raise
        raise SegmentRunError(f"invalid metrics for {run_paths.segment.name}: {exc}") from exc
    return metrics


def _mark_failed(run_paths: RunPaths) -> None:
    trial_dir = run_paths.trial_dir
    if trial_dir is None:
        return
    meta_path = trial_dir / "trial_meta.json"
    if meta_path.exists():
        update_segment(trial_dir, run_paths.segment.name, "failed")

