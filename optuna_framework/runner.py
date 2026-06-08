"""Run rendered ``runCombo.py`` configs and validate resulting metrics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from optuna_framework.metrics_parser import parse_window_metrics
from optuna_framework.paths import get_repo_root
from optuna_framework.specs import RunPaths
from optuna_framework.trial_meta import update_window


class InferenceRunError(RuntimeError):
    """Raised when one inference subprocess fails or emits invalid metrics."""


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


def run_inference(run_paths: RunPaths) -> object:
    """Execute one rendered inference run and return parsed scoring metrics."""

    cmd = build_run_command(run_paths.config_path)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
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
            raise InferenceRunError(_format_failed_run_message(run_paths, cmd, proc.returncode))
        if not run_paths.pnl_summary_path.exists() or run_paths.pnl_summary_path.stat().st_size <= 0:
            raise InferenceRunError(f"missing or empty pnl_summary.csv: {run_paths.pnl_summary_path}")
        metrics = parse_window_metrics(
            run_paths.pnl_summary_path,
            run_start_ds=run_paths.run_start_ds,
            run_end_ds=run_paths.run_end_ds,
            score_start_ds=run_paths.score_start_ds,
            score_end_ds=run_paths.score_end_ds,
        )
    except Exception as exc:
        _mark_failed(run_paths)
        if isinstance(exc, InferenceRunError):
            raise
        raise InferenceRunError(f"invalid metrics for {run_paths.run_dir}: {exc}") from exc
    return metrics


def _format_failed_run_message(run_paths: RunPaths, cmd: list[str], returncode: int) -> str:
    """Build an actionable error message for a failed inference subprocess."""

    details = [
        f"runCombo failed for {run_paths.run_dir} with returncode={returncode}",
        f"command: {command_to_string(cmd)}",
        f"config: {run_paths.config_path}",
        f"stdout_log: {run_paths.stdout_path}",
        f"stderr_log: {run_paths.stderr_path}",
    ]
    stderr_tail = _tail_text(run_paths.stderr_path)
    if stderr_tail:
        details.append("stderr_tail:\n" + stderr_tail)
    stdout_tail = _tail_text(run_paths.stdout_path)
    if stdout_tail:
        details.append("stdout_tail:\n" + stdout_tail)
    return "\n".join(details)


def _tail_text(path: Path, max_lines: int = 40, max_chars: int = 6000) -> str:
    """Return the last non-empty lines from a UTF-8-ish text file."""

    if not path.exists() or path.stat().st_size <= 0:
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()[-max_lines:]
    tail = "\n".join(lines).strip()
    if len(tail) > max_chars:
        tail = "..." + tail[-max_chars:]
    return tail


def _mark_failed(run_paths: RunPaths) -> None:
    trial_dir = run_paths.trial_dir
    if trial_dir is None:
        return
    meta_path = trial_dir / "trial_meta.json"
    if meta_path.exists():
        update_window(trial_dir, "failed")
