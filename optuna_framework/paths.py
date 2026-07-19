"""Centralized path construction for Optuna inference runs."""

from __future__ import annotations

from pathlib import Path

from optuna_framework.specs import RunPaths


def get_repo_root() -> Path:
    """Return the repository root that contains ``optuna_framework``."""

    return Path(__file__).resolve().parents[1]


def resolve_study_root(study_root: str | Path | None, study_name: str) -> Path:
    """Resolve a study root to an absolute path without creating it."""

    if study_root is None:
        root = get_repo_root() / "optuna_runs" / study_name
    else:
        root = Path(study_root).expanduser()
        if not root.is_absolute():
            root = get_repo_root() / root
    return root.resolve()


def build_baseline_run_paths(
    study_root: str | Path,
    run_window: tuple[int, int],
    score_window: tuple[int, int] | None = None,
) -> RunPaths:
    """Build absolute paths for the single full-window baseline run."""

    root = Path(study_root).expanduser().resolve()
    return _build_run_paths(
        study_root=root,
        run_dir=root / "baseline" / "full_run",
        run_window=run_window,
        score_window=score_window,
        snaptime="baseline_full_run",
        kind="baseline",
        trial_number=None,
    )


def build_trial_run_paths(
    study_root: str | Path,
    trial_number: int,
    run_window: tuple[int, int],
    score_window: tuple[int, int] | None = None,
) -> RunPaths:
    """Build absolute paths for one Phase A trial inference run."""

    root = Path(study_root).expanduser().resolve()
    trial_label = f"trial_{int(trial_number):05d}"
    return _build_run_paths(
        study_root=root,
        run_dir=root / "trials" / trial_label,
        run_window=run_window,
        score_window=score_window,
        snaptime=trial_label,
        kind="trial",
        trial_number=int(trial_number),
    )


def build_named_run_paths(
    study_root: str | Path,
    subdir: str | Path,
    kind: str,
    run_window: tuple[int, int],
    score_window: tuple[int, int] | None = None,
    snaptime: str | None = None,
    trial_number: int | None = None,
) -> RunPaths:
    """Build paths for Phase B/C and seed sanity runs."""

    root = Path(study_root).expanduser().resolve()
    run_dir = Path(subdir).expanduser()
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    if snaptime is None:
        snaptime = run_dir.name
    return _build_run_paths(
        study_root=root,
        run_dir=run_dir,
        run_window=run_window,
        score_window=score_window,
        snaptime=snaptime,
        kind=kind,
        trial_number=trial_number,
    )


def _build_run_paths(
    study_root: Path,
    run_dir: Path,
    run_window: tuple[int, int],
    score_window: tuple[int, int] | None,
    snaptime: str,
    kind: str,
    trial_number: int | None,
) -> RunPaths:
    run_start_ds, run_end_ds = _coerce_window(run_window, "run_window")
    if score_window is None:
        score_start_ds, score_end_ds = run_start_ds, run_end_ds
    else:
        score_start_ds, score_end_ds = _coerce_window(score_window, "score_window")
    run_dir = run_dir.resolve()
    output_root = (run_dir / "output").resolve()
    checkpoint_root = (output_root / "checkpoints").resolve()
    return RunPaths(
        study_root=study_root,
        run_dir=run_dir,
        config_path=(run_dir / "config.xml").resolve(),
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        pnl_summary_path=(output_root / "backtest" / "pnl_summary.csv").resolve(),
        stdout_path=(run_dir / "run.stdout.log").resolve(),
        stderr_path=(run_dir / "run.stderr.log").resolve(),
        params_path=(run_dir / "params.json").resolve(),
        resolved_meta_path=(run_dir / "resolved_meta.json").resolve(),
        snaptime=snaptime,
        kind=kind,
        trial_number=trial_number,
        run_start_ds=run_start_ds,
        run_end_ds=run_end_ds,
        score_start_ds=score_start_ds,
        score_end_ds=score_end_ds,
    )


def _coerce_window(window: tuple[int, int], name: str) -> tuple[int, int]:
    if len(window) != 2:
        raise ValueError(f"{name} must be a two-item (start_ds, end_ds) tuple")
    start_ds, end_ds = int(window[0]), int(window[1])
    if start_ds > end_ds:
        raise ValueError(f"{name} start_ds must be <= end_ds: {start_ds}-{end_ds}")
    return start_ds, end_ds
