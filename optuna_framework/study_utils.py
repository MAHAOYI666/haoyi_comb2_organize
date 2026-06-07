"""Shared utilities for Optuna scripts.

This module intentionally delays imports of optional dependencies such as
``optuna``, ``psutil``, and Plotly-backed Optuna visualizations until the
specific function that needs them is called.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from optuna_framework.runner import InferenceRunError


def missing_dependency_message(package: str) -> str:
    """Return the standard optional dependency error message."""

    return f"Missing dependency: {package}. Please install with `pip install {package}` before running this script."


def maybe_enqueue_baseline(study: Any, baseline_params: dict[str, Any]) -> bool:
    """Enqueue baseline params only when the study has no existing active/history trial."""

    try:
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError(missing_dependency_message("optuna")) from exc
    states = (
        TrialState.COMPLETE,
        TrialState.RUNNING,
        TrialState.WAITING,
        TrialState.PRUNED,
        TrialState.FAIL,
    )
    existing = [trial for trial in study.trials if trial.state in states]
    if len(existing) == 0:
        study.enqueue_trial(dict(baseline_params))
        return True
    return False


def create_study(
    study_name: str,
    storage: str,
    smoke: bool = False,
) -> Any:
    """Create or load an Optuna study with formal or smoke sampler settings."""

    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise RuntimeError(missing_dependency_message("optuna")) from exc

    if smoke:
        sampler = TPESampler(n_startup_trials=2, multivariate=True, group=True, seed=42)
        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=2, interval_steps=1)
    else:
        sampler = TPESampler(n_startup_trials=24, multivariate=True, group=True, seed=42)
        pruner = MedianPruner(n_startup_trials=24, n_warmup_steps=2, interval_steps=1)
    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )


def optimize_study(study: Any, objective: Any, n_trials: int, callbacks: list[Any] | None = None) -> None:
    """Run Optuna optimization while catching inference failures."""

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(InferenceRunError,), callbacks=callbacks)
    except KeyboardInterrupt:
        raise


def completed_history_count(study: Any) -> int:
    """Count trials that should reduce remaining requested trial budget."""

    try:
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError(missing_dependency_message("optuna")) from exc
    states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    return sum(1 for trial in study.trials if trial.state in states)


def append_resource_metric(study_root: Path, trial: Any) -> None:
    """Append a lightweight RSS sample for a completed trial."""

    try:
        import psutil
    except ImportError:
        print("Warning: psutil is not installed; skipping resource_metrics.csv update.")
        return
    reports_dir = Path(study_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "resource_metrics.csv"
    process = psutil.Process()
    rss_mb = process.memory_info().rss / 1024 / 1024
    row = {
        "trial_number": getattr(trial, "number", ""),
        "state": str(getattr(trial, "state", "")),
        "rss_mb": f"{rss_mb:.3f}",
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_study_reports(study: Any, study_root: Path) -> None:
    """Write trials and scoring metric CSV reports from Optuna and trial_meta files."""

    reports_dir = Path(study_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": str(trial.state).split(".")[-1],
            "value": trial.value,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        trial_rows.append(row)
    pd.DataFrame(trial_rows).to_csv(reports_dir / "trials.csv", index=False)

    scoring_rows = []
    for meta_path in sorted((Path(study_root) / "trials").glob("trial_*/trial_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = meta.get("scoring_window", {})
        row = {
            "trial_number": meta.get("trial_number"),
            "state": meta.get("state"),
            "objective": meta.get("objective"),
            "sharpe_idx": metrics.get("sharpe_idx"),
            "dd_li": metrics.get("dd_li"),
            "li_ret": metrics.get("li_ret"),
            "ret": metrics.get("ret"),
            "days": metrics.get("days"),
        }
        scoring_rows.append(row)
    pd.DataFrame(scoring_rows).to_csv(reports_dir / "scoring_metrics.csv", index=False)
    _write_top10(reports_dir / "trials.csv", reports_dir / "top10.csv")


def export_optuna_visualizations(study: Any, study_root: Path) -> None:
    """Export optional Optuna visualization HTML files."""

    try:
        from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances
    except Exception:
        print("Warning: plotly/optuna visualization is unavailable; skipping HTML exports.")
        return
    reports_dir = Path(study_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_param_importances(study).write_html(str(reports_dir / "param_importance.html"))
        plot_parallel_coordinate(study).write_html(str(reports_dir / "parallel_coordinate.html"))
        plot_optimization_history(study).write_html(str(reports_dir / "optimization_history.html"))
    except Exception as exc:
        print(f"Warning: failed to export Optuna visualizations: {exc}")


def cleanup_bad_trial_artifacts(trial_dir: Path) -> None:
    """Remove heavyweight artifacts while preserving configs, metrics, and logs."""

    for checkpoint_dir in [trial_dir / "checkpoints", *trial_dir.glob("*/checkpoints")]:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
    for alpha_history in [trial_dir / "output" / "alpha_history.pt", *trial_dir.glob("*/output/alpha_history.pt")]:
        alpha_history.unlink(missing_ok=True)


def _write_top10(trials_csv: Path, output_csv: Path) -> None:
    if not trials_csv.exists():
        return
    df = pd.read_csv(trials_csv)
    if df.empty or "value" not in df.columns:
        df.head(0).to_csv(output_csv, index=False)
        return
    ranked = df[pd.to_numeric(df["value"], errors="coerce") != -10.0].copy()
    ranked["value"] = pd.to_numeric(ranked["value"], errors="coerce")
    ranked = ranked.dropna(subset=["value"]).sort_values("value", ascending=False).head(10)
    ranked.to_csv(output_csv, index=False)
