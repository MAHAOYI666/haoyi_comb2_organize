"""Centralized path construction for Optuna runs."""

from __future__ import annotations

from pathlib import Path

from optuna_framework.specs import RunPaths, SegmentSpec


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


def build_trial_run_paths(study_root: str | Path, trial_number: int, segment: SegmentSpec) -> RunPaths:
    """Build unique absolute paths for one trial/segment pair."""

    root = Path(study_root).expanduser().resolve()
    trial_label = f"trial_{int(trial_number):05d}"
    segment_dir = root / "trials" / trial_label / segment.name
    snaptime = f"{trial_label}_{segment.name}"
    return _build_run_paths(
        study_root=root,
        segment_dir=segment_dir,
        segment=segment,
        snaptime=snaptime,
        kind="trial",
        trial_number=int(trial_number),
    )


def build_baseline_run_paths(study_root: str | Path, segment: SegmentSpec) -> RunPaths:
    """Build absolute paths for one baseline/segment pair."""

    root = Path(study_root).expanduser().resolve()
    segment_dir = root / "baseline" / segment.name
    snaptime = f"baseline_{segment.name}"
    return _build_run_paths(
        study_root=root,
        segment_dir=segment_dir,
        segment=segment,
        snaptime=snaptime,
        kind="baseline",
        trial_number=None,
    )


def build_named_run_paths(
    study_root: str | Path,
    base_dir: str | Path,
    segment: SegmentSpec,
    snaptime: str,
    kind: str,
    trial_number: int | None = None,
) -> RunPaths:
    """Build paths for phase-B/C and seed sanity runs."""

    root = Path(study_root).expanduser().resolve()
    return _build_run_paths(
        study_root=root,
        segment_dir=Path(base_dir).expanduser().resolve(),
        segment=segment,
        snaptime=snaptime,
        kind=kind,
        trial_number=trial_number,
    )


def _build_run_paths(
    study_root: Path,
    segment_dir: Path,
    segment: SegmentSpec,
    snaptime: str,
    kind: str,
    trial_number: int | None,
) -> RunPaths:
    segment_dir = segment_dir.resolve()
    output_root = (segment_dir / "output").resolve()
    checkpoint_root = (segment_dir / "checkpoints").resolve()
    return RunPaths(
        study_root=study_root,
        segment_dir=segment_dir,
        config_path=(segment_dir / "config.xml").resolve(),
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        pnl_summary_path=(output_root / "backtest" / "pnl_summary.csv").resolve(),
        stdout_path=(segment_dir / "run.stdout.log").resolve(),
        stderr_path=(segment_dir / "run.stderr.log").resolve(),
        params_path=(segment_dir / "params.json").resolve(),
        resolved_meta_path=(segment_dir / "resolved_meta.json").resolve(),
        snaptime=snaptime,
        segment=segment,
        kind=kind,
        trial_number=trial_number,
    )

