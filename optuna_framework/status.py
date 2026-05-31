"""Small status-printing helpers for Optuna orchestration."""

from __future__ import annotations

from datetime import datetime


def log_status(*parts: object) -> None:
    """Print one flushed, timestamped Optuna status line."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[OPTUNA]", timestamp, *parts, flush=True)
