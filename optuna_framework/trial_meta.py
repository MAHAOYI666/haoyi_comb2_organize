"""Atomic ``trial_meta.json`` state management."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


META_FILENAME = "trial_meta.json"


def init_trial_meta(
    trial_dir: str | Path,
    trial_number: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Initialize ``trial_meta.json`` for a trial."""

    meta = {
        "trial_number": int(trial_number),
        "state": "running",
        "started_at": _now(),
        "ended_at": None,
        "params": dict(params),
        "scoring_window": {"state": "pending"},
        "objective": None,
        "hard_filter_triggered": False,
    }
    write_trial_meta(trial_dir, meta)
    return meta


def read_trial_meta(trial_dir: str | Path) -> dict[str, Any]:
    """Read ``trial_meta.json`` from a trial directory."""

    return json.loads(_meta_path(trial_dir).read_text(encoding="utf-8"))


def write_trial_meta(trial_dir: str | Path, meta: dict[str, Any]) -> None:
    """Atomically write ``trial_meta.json`` using tmp file + ``os.replace``."""

    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    target = _meta_path(trial_dir)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def update_window(
    trial_dir: str | Path,
    state: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update the single scoring-window state and optional metrics."""

    meta = read_trial_meta(trial_dir)
    window = dict(meta.get("scoring_window", {}))
    window["state"] = state
    if metrics:
        window.update(metrics)
    meta["scoring_window"] = window
    write_trial_meta(trial_dir, meta)
    return meta


def update_trial_state(
    trial_dir: str | Path,
    state: str,
    objective: float | None = None,
    hard_filter_triggered: bool | None = None,
) -> dict[str, Any]:
    """Update the top-level trial state."""

    meta = read_trial_meta(trial_dir)
    meta["state"] = state
    if state in {"complete", "pruned", "failed"}:
        meta["ended_at"] = _now()
    if objective is not None:
        meta["objective"] = float(objective)
    if hard_filter_triggered is not None:
        meta["hard_filter_triggered"] = bool(hard_filter_triggered)
    write_trial_meta(trial_dir, meta)
    return meta


def _meta_path(trial_dir: str | Path) -> Path:
    return Path(trial_dir) / META_FILENAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
