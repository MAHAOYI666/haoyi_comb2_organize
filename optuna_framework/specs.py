"""Typed runtime path records shared by the Optuna search framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """All filesystem paths and windows owned by one isolated inference run."""

    study_root: Path
    run_dir: Path
    config_path: Path
    output_root: Path
    checkpoint_root: Path
    pnl_summary_path: Path
    stdout_path: Path
    stderr_path: Path
    params_path: Path
    resolved_meta_path: Path
    snaptime: str
    kind: str
    trial_number: int | None
    run_start_ds: int
    run_end_ds: int
    score_start_ds: int
    score_end_ds: int

    @property
    def trial_dir(self) -> Path | None:
        """Return the owning trial directory, if this is a Phase A trial run."""

        if self.trial_number is None:
            return None
        return self.run_dir

    @property
    def run_window(self) -> tuple[int, int]:
        """Return the rendered strategy window."""

        return (self.run_start_ds, self.run_end_ds)

    @property
    def score_window(self) -> tuple[int, int]:
        """Return the objective/scoring window."""

        return (self.score_start_ds, self.score_end_ds)
