"""Typed specifications shared by the Optuna search framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SegmentSpec:
    """A dated evaluation segment for tuning, holdout, or baseline-only runs."""

    name: str
    role: str
    start_ds: int
    end_ds: int


@dataclass(frozen=True)
class RunPaths:
    """All filesystem paths owned by one baseline/trial segment run."""

    study_root: Path
    segment_dir: Path
    config_path: Path
    output_root: Path
    checkpoint_root: Path
    pnl_summary_path: Path
    stdout_path: Path
    stderr_path: Path
    params_path: Path
    resolved_meta_path: Path
    snaptime: str
    segment: SegmentSpec
    kind: str
    trial_number: int | None = None

    @property
    def trial_dir(self) -> Path | None:
        """Return the owning trial directory, if this is a trial run."""

        if self.trial_number is None:
            return None
        return self.segment_dir.parent


@dataclass(frozen=True)
class StudySpec:
    """High-level search configuration that can be swapped per model adapter."""

    name: str
    baseline_config_path: Path
    adapter_name: str | None
    tuning_segments: tuple[SegmentSpec, ...]
    holdout_segments: tuple[SegmentSpec, ...]
    baseline_only_segments: tuple[SegmentSpec, ...]
    n_trials: int = 60
    fixed_overrides: dict[str, Any] = field(default_factory=dict)
    plugin_path: Path | None = None
    phase_b_seeds: tuple[int, ...] = (42, 43, 44)

    def baseline_segments(self) -> tuple[SegmentSpec, ...]:
        """Return all segments that baseline evaluation should render."""

        return self.tuning_segments + self.holdout_segments + self.baseline_only_segments

    def segment_by_name(self, name: str) -> SegmentSpec:
        """Find a segment by name."""

        for segment in self.all_segments():
            if segment.name == name:
                return segment
        raise KeyError(f"unknown segment: {name}")

    def all_segments(self) -> tuple[SegmentSpec, ...]:
        """Return every segment declared by this study."""

        return self.baseline_segments()

    def iter_segments(self, roles: Iterable[str]) -> tuple[SegmentSpec, ...]:
        """Return segments whose role is in ``roles``."""

        role_set = set(roles)
        return tuple(segment for segment in self.all_segments() if segment.role in role_set)
